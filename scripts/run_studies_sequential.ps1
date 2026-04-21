param(
    [string]$OutputRoot = "outputs\full10_run",
    [string]$Profile = "full10",
    [string]$Device = "cuda",
    [int]$ParallelWorkers = 0,
    [int]$GpuIndex = 0,
    [double]$MinFreeVramMb = 500,
    [double]$SafetyMarginMb = 500,
    [double]$PerProcessOverheadMb = 256,
    [double]$PollIntervalSec = 2,
    [double]$LaunchSettleSec = 1
)

$ErrorActionPreference = "Stop"
Set-Location "D:\Projects\transformer-generalization"

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$Command
    )

    Write-Host "[$(Get-Date -Format o)] START $Label"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    Write-Host "[$(Get-Date -Format o)] END $Label"
}

function Write-SchedulerEvent {
    param(
        [string]$Path,
        [hashtable]$Payload
    )

    $record = [ordered]@{
        ts_utc = [DateTime]::UtcNow.ToString("o")
    }
    foreach ($key in $Payload.Keys) {
        $record[$key] = $Payload[$key]
    }
    ($record | ConvertTo-Json -Compress) + "`n" | Out-File -FilePath $Path -Encoding utf8 -Append
}

function Get-FreeVramMb {
    param([int]$GpuIndex)

    $raw = & nvidia-smi "--query-gpu=memory.free,memory.total" "--format=csv,noheader,nounits" "--id=$GpuIndex"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) {
        throw "failed to query GPU memory with nvidia-smi"
    }
    $parts = $raw.Trim().Split(",")
    return @{
        free = [double]$parts[0].Trim()
        total = [double]$parts[1].Trim()
    }
}

function Estimate-JobVramMb {
    param([pscustomobject]$Job)

    $run = $Job.run_config
    $task = $Job.task
    $batchSize = [int]$run.batch_size
    $fullBatch = [bool]$run.full_batch
    $modelType = [string]$run.model_type
    $objective = [string]$run.objective
    $perProcessOverhead = [double]$PerProcessOverheadMb

    if ($task.kind -eq "single_operator") {
        $modulus = [int]$task.modulus
        $seqLen = 3
        $targetVocab = $modulus
        $trainSize = [int](($modulus * $modulus) * [double]$task.train_fraction)
    } else {
        $modulus = [int]$task.modulus
        $seqLen = 4
        $targetVocab = [Math]::Max(([int]$task.add_offset + $modulus - 1), ([int]$task.mul_offset + $modulus - 1)) + 1
        $trainSize = [int]((2 * $modulus * $modulus) * [double]$task.train_fraction)
    }

    $effectiveBatch = if ($fullBatch) { $trainSize } else { [Math]::Min($batchSize, $trainSize) }

    if ($modelType -eq "transformer") {
        $dModel = 128.0
        $dMlp = 512.0
        $activationMb = 4.0 * $effectiveBatch * $seqLen * (6 * $dModel + 2 * $dMlp + $targetVocab) / (1024.0 * 1024.0)
        $rlExtra = 0.0
        if ($objective -eq "grpo" -and $run.grpo) {
            $nSamples = [int]$run.grpo.n_samples
            $rlExtra = 4.0 * $effectiveBatch * $nSamples * 4 / (1024.0 * 1024.0)
        }
        return $perProcessOverhead + [Math]::Max(150.0, $activationMb + $rlExtra + 64.0)
    }

    $hiddenDim = [double]$run.mlp_hidden_dim
    $activationMb = 4.0 * $effectiveBatch * (3 * $modulus + $hiddenDim) / (1024.0 * 1024.0)
    $rlExtra = 0.0
    if ($objective -eq "ppo" -and $run.ppo) {
        $nSamples = [int]$run.ppo.n_samples
        $rlExtra = 4.0 * $effectiveBatch * $nSamples * 6 / (1024.0 * 1024.0)
    }
    return $perProcessOverhead + [Math]::Max(150.0, $activationMb + $rlExtra + 64.0)
}

function Start-JobProcess {
    param(
        [string]$ManifestPath,
        [int]$JobIndex,
        [string]$OutputDir
    )

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $stdoutPath = Join-Path $OutputDir "launcher.stdout.log"
    $stderrPath = Join-Path $OutputDir "launcher.stderr.log"
    return Start-Process -FilePath ".\.venv\Scripts\python.exe" `
        -ArgumentList "-u",".\scripts\run_experiment_job.py","--manifest",$ManifestPath,"--job-index","$JobIndex" `
        -WorkingDirectory "D:\Projects\transformer-generalization" `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
}

function Invoke-ManifestScheduler {
    param(
        [string]$StudyName,
        [string]$ManifestPath,
        [string]$SummaryPath
    )

    $eventPath = Join-Path (Split-Path $SummaryPath -Parent) "scheduler_events.jsonl"
    $parallelLimit = if ($ParallelWorkers -le 0) { [int]::MaxValue } else { $ParallelWorkers }
    $jobs = Get-Content $ManifestPath | ForEach-Object { $_ | ConvertFrom-Json }
    $pending = @()
    for ($i = 0; $i -lt $jobs.Count; $i++) {
        $pending += [pscustomobject]@{
            index = $i
            job = $jobs[$i]
            estimated_vram_mb = Estimate-JobVramMb -Job $jobs[$i]
        }
    }
    $pending = $pending | Sort-Object estimated_vram_mb -Descending

    $gpu = Get-FreeVramMb -GpuIndex $GpuIndex
    Write-SchedulerEvent -Path $eventPath -Payload @{
        event = "scheduler_start"
        gpu_index = $GpuIndex
        total_vram_mb = $gpu.total
        pending_jobs = $pending.Count
        max_parallel = if ($ParallelWorkers -le 0) { $null } else { $ParallelWorkers }
        min_free_vram_mb = $MinFreeVramMb
        safety_margin_mb = $SafetyMarginMb
        per_process_overhead_mb = $PerProcessOverheadMb
    }

    $running = @()
    while ($pending.Count -gt 0 -or $running.Count -gt 0) {
        $stillRunning = @()
        foreach ($item in $running) {
            if ($item.process.HasExited) {
                Write-SchedulerEvent -Path $eventPath -Payload @{
                    event = "finish"
                    job_index = $item.job_index
                    return_code = $item.process.ExitCode
                    output_dir = $item.output_dir
                    pid = $item.process.Id
                }
                if ($item.process.ExitCode -ne 0) {
                    throw "$StudyName job $($item.job_index) failed with exit code $($item.process.ExitCode)"
                }
            } else {
                $stillRunning += $item
            }
        }
        $running = $stillRunning

        $launchedAny = $false
        while ($pending.Count -gt 0 -and $running.Count -lt $parallelLimit) {
            $gpu = Get-FreeVramMb -GpuIndex $GpuIndex
            $capacity = $gpu.free - $SafetyMarginMb
            if ($capacity -lt $MinFreeVramMb) {
                Write-SchedulerEvent -Path $eventPath -Payload @{
                    event = "wait"
                    reason = "below_min_free_threshold"
                    free_vram_mb = $gpu.free
                    total_vram_mb = $gpu.total
                    capacity_mb = $capacity
                    running_jobs = $running.Count
                    pending_jobs = $pending.Count
                }
                break
            }

            $selected = $pending | Where-Object { $_.estimated_vram_mb -le $capacity } | Select-Object -First 1
            if (-not $selected) {
                Write-SchedulerEvent -Path $eventPath -Payload @{
                    event = "wait"
                    reason = "no_job_fits_current_free_vram"
                    free_vram_mb = $gpu.free
                    total_vram_mb = $gpu.total
                    capacity_mb = $capacity
                    running_jobs = $running.Count
                    pending_jobs = $pending.Count
                }
                break
            }

            $outputDir = [string]$selected.job.run_config.output_dir
            $process = Start-JobProcess -ManifestPath $ManifestPath -JobIndex ([int]$selected.index) -OutputDir $outputDir
            Write-SchedulerEvent -Path $eventPath -Payload @{
                event = "launch"
                job_index = $selected.index
                estimated_vram_mb = $selected.estimated_vram_mb
                output_dir = $outputDir
                pid = $process.Id
            }
            $running += [pscustomobject]@{
                process = $process
                job_index = [int]$selected.index
                output_dir = $outputDir
            }
            $pending = @($pending | Where-Object { $_.index -ne $selected.index })
            $launchedAny = $true
            Start-Sleep -Seconds $LaunchSettleSec
        }

        if ($pending.Count -gt 0 -or $running.Count -gt 0) {
            if (-not $launchedAny) {
                Start-Sleep -Seconds $PollIntervalSec
            }
        }
    }

    Write-SchedulerEvent -Path $eventPath -Payload @{ event = "scheduler_end" }
    Invoke-Step "$StudyName aggregate" @(
        ".\.venv\Scripts\python.exe",
        "-u",
        ".\scripts\aggregate_manifest_results.py",
        "--manifest",
        $ManifestPath,
        "--summary-out",
        $SummaryPath
    )
}

function Run-Study {
    param(
        [string]$StudyName,
        [string]$ScriptPath
    )

    $studyRoot = Join-Path $OutputRoot $StudyName
    $manifestPath = Join-Path $studyRoot "manifest.jsonl"
    $summaryPath = Join-Path $studyRoot "summary.csv"

    Invoke-Step "$StudyName manifest" @(
        ".\.venv\Scripts\python.exe",
        "-u",
        $ScriptPath,
        "--profile",
        $Profile,
        "--output-root",
        $studyRoot,
        "--device",
        $Device,
        "--manifest-out",
        $manifestPath,
        "--manifest-only"
    )

    Write-Host "[$(Get-Date -Format o)] START $StudyName scheduler"
    Invoke-ManifestScheduler -StudyName $StudyName -ManifestPath $manifestPath -SummaryPath $summaryPath
    Write-Host "[$(Get-Date -Format o)] END $StudyName scheduler"
}

Run-Study -StudyName "study1_loss_vs_rl" -ScriptPath ".\scripts\run_loss_vs_rl.py"
Run-Study -StudyName "study2_fake_labels" -ScriptPath ".\scripts\run_fake_labels.py"
Run-Study -StudyName "study3_range_transfer" -ScriptPath ".\scripts\run_range_transfer.py"
