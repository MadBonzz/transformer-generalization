# Grokking Notes Compiled from This Chat

## Paper shorthand used in this chat

- **Original grokking paper**: *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (Power et al., 2022)
- **Neel Nanda paper**: *Progress Measures for Grokking via Mechanistic Interpretability* (Nanda et al., 2023)
- **Wrong data paper**: *To Grok or Not to Grok: Disentangling Generalization and Memorization on Corrupted Algorithmic Datasets* (2023)
- **Double/triple descent paper**: *Deep Double Descent: Where Bigger Models and More Data Hurt* (2020)
- **Omnigrok paper**: *Omnigrok: Grokking Beyond Algorithmic Data* (2022)
- **Clock/Pizza paper**: the paper discussed in the Quanta article saying that different runs can learn different algorithms for the same task

---

# 1. Core picture of grokking

The original grokking result is:

- train accuracy becomes very high very early,
- validation/test accuracy stays poor for a long time,
- then test accuracy suddenly jumps much later.

This is most cleanly shown on tiny synthetic algorithmic datasets such as modular arithmetic and finite-group operations.

Important takeaways:

- **Smaller datasets** can still eventually generalize, but may need much longer optimization.
- **Weight decay** strongly helps grokking in the original setup.
- The late jump in generalization is one of the main signatures of the phenomenon.

---

# 2. Input format and sequence length

## Original grokking paper

In the original paper, equations are tokenized like:

`x  op  y  =  answer`

So the full sequence is effectively:

1. first operand
2. operator
3. second operand
4. equals sign
5. answer

### Conclusion
- **Sequence length = 5**

This was important because I initially asked whether the max seq len was effectively 4; the answer is **no** for the original paper. It is **5**.

## Neel Nanda paper

In Neel Nanda’s modular-addition setup, the operation is fixed, so the operator token is omitted. The model sees something like:

`a  b  =`

and predicts the answer from the equals position.

### Conclusion
- actual fed-in context is **3 tokens**
- if one informally counts the predicted answer slot too, one could loosely think of it as 4 positions, but the actual input is **3**

## General conclusion

For the **classic binary-operator modular arithmetic family**, a transformer that supports **sequence length up to 5** is enough for the standard experiments.

But this only applies to the tiny symbolic arithmetic setups. Other tasks in related papers can require more context.

---

# 3. Are these datasets artificial?

Yes.

These grokking datasets are generally **synthetically generated** from known rules.

Examples:
- modular addition
- modular subtraction
- modular division
- polynomial-like modular operators
- finite-group operations

This is useful because:
- the ground-truth function is known exactly,
- one can generate the full dataset exhaustively,
- memorization vs rule-learning can be studied very cleanly.

So yes, the dataset is usually created artificially rather than scraped or collected from the world.

---

# 4. Typical model sizes used in grokking work

## Original grokking paper

Typical transformer:
- decoder-only transformer
- **2 layers**
- **width 128**
- **4 attention heads**
- about **4e5 non-embedding parameters**

## Neel Nanda paper

Mainline interpretable transformer:
- **1 layer**
- **d_model = 128**
- **4 heads**
- **head dimension = 32**
- **MLP hidden size = 512**
- deliberately small for mechanistic interpretability

The point here is that the canonical papers mostly use **small models on purpose**, because interpretability becomes much easier.

## Beyond 2-layer transformers?

Yes, later work explores deeper models too, but the most famous mechanistic grokking papers are usually in the 1–2 layer regime.

---

# 5. Can these models run on 6 GB VRAM?

Yes, for the standard tiny grokking setups, **6 GB VRAM should be enough**.

Why:
- model sizes are very small,
- sequence lengths are tiny (3–5),
- these are toy setups, not large LLMs.

Caveats:
- very large batch sizes can still increase memory use,
- many parallel runs / extensive logging / many checkpoints can also add overhead,
- scaling depth/width a lot would change the answer.

But for **original-paper-style** and **Neel-style** modular grokking experiments, 6 GB VRAM should be comfortable.

---

# 6. What is the Neel Nanda paper saying mechanistically?

The Neel Nanda paper argues that the model learns a **Fourier-based circuit** for modular addition.

Rough idea:
- numbers mod p naturally live on a cycle,
- the network learns sine/cosine-like features over that cycle,
- addition becomes something like combining angular information,
- internal structure can be described in Fourier terms.

The paper decomposes grokking-like behavior into smoother hidden stages:

1. **memorization**
2. **circuit formation**
3. **cleanup**

The visible sudden jump in test accuracy is not necessarily because everything changed instantly at once, but because a generalizing circuit gradually formed and eventually dominated.

---

# 7. Clock vs Pizza and “different algorithms for the same task”

The Quanta article discussed a follow-up paper saying that even when models are trained on the **same task**, they can end up learning **different internal algorithms**.

This is important because it means:
- same architecture,
- same data,
- same task,
- different initialization / hyperparameters,

can still lead to **different mechanistic solutions**.

Examples mentioned:
- **Clock-style / Fourier-style** solution
- **Pizza-style** solution

So even in toy tasks, there may not be a single unique internal algorithm that every model converges to.

---

# 8. How do researchers infer what algorithm or equation a model is learning?

There is a methodology, but it is **not a fully automatic, universal recipe**.

## General workflow

### Step 1: exploit task structure
For modular arithmetic, the natural mathematical basis is often the **discrete Fourier basis**.

### Step 2: form a circuit hypothesis
Example:
- perhaps the model is representing numbers using sine/cosine components,
- perhaps it combines them according to a trig-like rule that corresponds to modular addition.

### Step 3: inspect weights and activations
Researchers inspect:
- embeddings,
- attention heads,
- MLP neurons,
- logits / unembedding directions,
- activity in specific subspaces.

### Step 4: do causal interventions
This is crucial.

They use things like:
- ablations,
- zeroing components,
- pruning,
- activation patching,
- intervening in Fourier subspaces,
- checking whether the behavior changes as predicted.

## Important conclusion

This is **reproducible as a research workflow**, but it is **not guaranteed to yield one unique correct explanation** for every model.

So:

- there **is** a methodology,
- it often works best in tiny toy models,
- but it is **not** a push-button universally reliable pipeline.

That is also why different groups can analyze similar small models and still disagree.

---

# 9. Wrong data paper: what exactly does it show?

This was one of the most important clarifications in the chat.

There are **two related but different findings**:

## A. Simultaneous memorization and generalization
The model can:
- memorize corrupted labels for corrupted training points,
- while also generalizing correctly on clean/test inputs.

This is the **coexistence** regime.

## B. Inversion / recovery of true labels
With sufficient regularization, the model can later start predicting the **true label even for corrupted training points**.

This means the model no longer follows the fake label for those examples.

So the Quanta article’s statement that the model can eventually output the correct answer even for the wrong examples is also true.

### Reconciliation
Both are true, but they describe **different phases/regimes**.

- coexistence = memorization + generalization simultaneously
- inversion = true rule takes over even on corrupted points

---

# 10. How is corruption done in the wrong data paper?

The corruption is done by taking a subset of training examples and **replacing their labels** with random labels.

Important detail:
- this is a **replacement**, not an addition.

So if a clean example was:

`(a, b) -> c`

and it is corrupted, it becomes:

`(a, b) -> d`

The training set does **not** normally contain both:
- `(a, b) -> c`
- `(a, b) -> d`

at the same time as two separate training rows.

So there are not intended to be duplicate clean-and-fake versions of the exact same input pair in the training data.

One subtle unresolved detail from the chat:
- if the fake label is sampled uniformly from all labels, there may be a tiny chance that the sampled replacement accidentally equals the original true label unless their implementation explicitly forbids this.
- This detail was not cleanly verified from the accessible paper text in the chat.

---

# 11. What regimes/phases does the wrong data paper describe?

Main regimes discussed:

1. **Memorization**
   - fit corrupted labels
   - poor true generalization

2. **Coexistence**
   - memorizes corrupted labels
   - also generalizes

3. **Partial inversion**
   - starts predicting true labels for some corrupted points

4. **Full inversion**
   - predicts true labels for all corrupted points
   - train accuracy drops to approximately the clean fraction, since the model is no longer obeying the fake labels

There can also be failure/confusion regimes depending on regularization strength and optimization.

---

# 12. What controls / ablations does the wrong data paper use?

The paper does much more than just show a single result.

It varies:
- training data fraction
- corruption/noise fraction
- weight decay
- dropout
- BatchNorm / normalization-related conditions
- architecture variants

It also tries to identify separate components for:
- memorization
- generalization

## Mechanistic tools used there

A key idea is to use **IPR (inverse participation ratio)** in Fourier space to distinguish different neuron types.

Very roughly:
- some neurons are associated more with the periodic/generalizing rule,
- others are associated more with memorization.

## Causal tests

They do pruning experiments:
- pruning some neurons hurts memorization more,
- pruning others hurts generalization more.

This supports the claim that the network can contain distinct subsystems for memorization and rule-like generalization.

## Effect of regularizers

Different regularizers do different things:

- **weight decay** strongly pushes toward the generalizing solution and can lead to inversion
- **dropout** can also favor generalization, though with somewhat different dynamics
- **BatchNorm** does not necessarily remove memorizing units the same way; instead it can suppress or rebalance them

---

# 13. Fake label percentages: what we established and what remained uncertain

We verified:
- the wrong-data paper does scan over different corruption/noise levels
- it is not just a single fixed corruption percentage

However, in the chat we did **not fully recover the exact numeric grid of all corruption percentages** from the accessible rendering.

So the correct note is:

- yes, multiple corruption levels are tried
- no, we did not pin down the exact full list of percentages in the chat with high confidence

This should be marked as an unresolved detail if I later want to build an exact reproduction table.

---

# 14. Omnigrok and broader significance

Omnigrok argues that grokking-like behavior is not restricted only to tiny modular arithmetic tasks.

It proposes a broader picture involving weight norm and loss geometry:
- reduced training loss looks roughly L-shaped,
- test loss can look U-shaped when viewed against norm,
- the model can first move into an overfitting region and later into a “Goldilocks” region that generalizes better.

The paper claims grokking-like phenomena can appear across:
- algorithmic tasks
- image tasks
- language tasks
- molecular tasks

This suggests grokking might be part of a broader class of delayed-generalization behaviors.

---

# 15. Relation to double descent

The double/triple descent paper is not itself a grokking paper in the narrow toy-task sense, but it is relevant background.

Why:
- it also shows that generalization behavior can be very non-monotonic,
- bigger models / more data do not always improve test error monotonically,
- there can be multiple overfitting peaks.

So it helps situate grokking within a larger family of surprising generalization phenomena.

---

# 16. Practical experimental takeaway

If I wanted to run my own grokking experiments modeled after the classic papers, a practical default setup would be:

## Task
- modular addition (mod p), or another small synthetic binary operator

## Dataset
- exhaustively generate all pairs
- sample a training fraction
- test on held-out pairs

## Input format
- either original style: `a op b = answer` (seq len 5)
- or fixed-op style: `a b =` (context len 3)

## Model
- tiny decoder-only transformer
- 1 or 2 layers
- d_model around 128
- 4 heads
- MLP hidden around 512

## Training
- AdamW
- weight decay is important if I want classic grokking behavior to appear reliably

## Analysis
- track both train and validation accuracy over long training
- inspect representations
- use basis-aware analysis (e.g. Fourier basis for modular arithmetic)
- do causal tests, not just visualization

---

# 17. Important cautions

1. **Tiny sequence length does not mean broad reasoning.**  
   These setups compress the task into a single-step symbolic mapping.

2. **Interpretability is easier because the task is tiny and structured.**  
   It does not automatically scale to large language models.

3. **Different models can learn different internal algorithms for the same task.**  
   So I should not assume all grokked solutions look like Neel’s Fourier/clock story.

4. **There is a methodology for mechanistic explanation, but not a universal guaranteed recipe.**

---

# 18. What I can refer to later in this chat

You said that in this chat you may refer to:

- **the original grokking paper**
- **the Neel Nanda paper**
- **the wrong data paper**

I have been using those shorthands with the following mapping:

- original grokking paper = Power et al. 2022
- Neel Nanda paper = mechanistic/Fourier analysis paper
- wrong data paper = corrupted-label grokking paper

---

# 19. Most useful crisp answers from this chat

## Is max seq len 4 for all these papers?
No.
- original paper: **5**
- Neel setup: **3** input tokens
- not all papers even use sequence models

## Can I use a transformer with max seq len 5 for related experiments?
Yes, for the classic binary-operator modular arithmetic setups.

## Are the datasets artificial?
Yes, usually fully synthetic.

## Do people use deeper/larger models?
Sometimes yes, but the classic canonical papers mostly use tiny 1–2 layer models.

## Can I run these on 6 GB VRAM?
Yes, for the standard toy setups.

## Does the wrong-data paper show simultaneous memorization and generalization?
Yes.

## Does it also show that the model can eventually output correct answers even for corrupted entries?
Yes.

## Are both clean and fake labels for the same input included simultaneously?
Not in the intended setup; the label is replaced, not duplicated.

---

# 20. Open questions / unresolved details from this chat

These are things not pinned down with full certainty here:

- the exact complete list/grid of corruption percentages used in every wrong-data experiment
- whether the corrupted-label sampling explicitly forbids re-sampling the true label
- exact architecture grids for every appendix experiment across all papers

If I need a strict reproduction protocol later, these are the first things I should verify again directly from code or appendices.

---

# 21. Suggested next steps if I continue this project

If I want to work further on grokking after this chat, the most useful next moves would be:

1. reproduce modular-addition grokking with a tiny transformer
2. reproduce corrupted-label experiments
3. analyze Fourier structure of learned embeddings/logits
4. test whether different seeds learn clock-like vs pizza-like solutions
5. compare weight decay vs dropout vs normalization systematically
6. test whether the same interpretability workflow still works on slightly larger/deeper models

