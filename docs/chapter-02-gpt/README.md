# Chapter 2: Building a small GPT from scratch, explained

Companion docs for [`src/Chapter_2__build_GPT_from_scratch.ipynb`](../../src/Chapter_2__build_GPT_from_scratch.ipynb). They cover the last third of the notebook: the finished model, how it is trained, and how it generates text.

> A corrected copy, [`src/Chapter_2__build_GPT_from_scratch_fixed.ipynb`](../../src/Chapter_2__build_GPT_from_scratch_fixed.ipynb), fixes four of the bugs listed in the docs' "Things to watch for" sections: the `x-0.5` residual, attention scaling, generating in train mode, and the load cells. The cosmetic items in those sections are left as they were. Each change is marked `# FIX:`.

| # | Doc | Notebook section | What you'll understand |
|---|---|---|---|
| 1 | [The `GPTLanguageModel` class](01-gpt-language-model.md) | Part 12 | Each layer, the tensor shapes through a forward pass, where the 209,729 parameters are, and how the loss is computed |
| 2 | [Training](02-training.md) | Part 13 | `get_batch`, the 5-line loop, AdamW, dropout, `estimate_loss`, and how to read the loss curve |
| 3 | [Inference](03-inference.md) | `generate()`, end of Part 13, "Load the model" | Autoregressive sampling, context cropping, temperature/top-k, `eval()`/`no_grad()`, saving and loading |

## The big picture in three sentences

1. **The model** turns a sequence of character ids into, at every position, a score for each of the 65 possible next characters.
2. **Training** shows it random 32-character windows of Shakespeare, scores those guesses against the real next characters (cross-entropy), and adjusts all weights to do better. This is repeated 5,000 times.
3. **Inference** asks for the next character, samples one, appends it, and asks again.

## Figures

All figures are in [`img/`](img/) as standalone SVGs.

| | |
|---|---|
| [![](img/gpt-architecture.svg)](img/gpt-architecture.svg) | [![](img/transformer-block.svg)](img/transformer-block.svg) |
| **Model architecture & shapes** | **Inside one Block** |
| [![](img/next-token-targets.svg)](img/next-token-targets.svg) | [![](img/training-loop.svg)](img/training-loop.svg) |
| **Inputs vs targets** | **The training loop** |
| [![](img/loss-curve.svg)](img/loss-curve.svg) | [![](img/generate-loop.svg)](img/generate-loop.svg) |
| **Loss curve (real run)** | **The generate loop** |
| [![](img/sampling-temperature.svg)](img/sampling-temperature.svg) | |
| **Temperature** | |

## Where the numbers come from

Every number in these docs (parameter counts, losses, probabilities, sample text) was measured by running the notebook's own cells unchanged: the data cells, the Part 11–12 classes, and the Part 13 training cell. The run used one GPU and took about 2 minutes. The bigram baseline comes from the Part 7–8 cells. Your exact values will differ slightly, because every cell shares one random number generator. The trends will be the same.
