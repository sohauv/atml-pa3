# ATML PA3

Reproducible experiments for EE-5102/CS-6304 Programming Assignment 3.
Final experiment evidence is grouped by task under `results/`.

## Environment

Install the pinned dependencies with `python -m pip install -r requirements.txt`. The saved results can be inspected directly.

## Methodology

Task 1 uses the first 40,000 eligible MetaMathQA examples in source order, no
shuffle/truncation, seed 6304, Qwen3-0.6B non-thinking templates, and response/EOT
labels. Filtering uses NFKC, casefold, and collapsed whitespace on both query and
original question against GSM8K test questions. Dataset metadata records exact
upstream revisions, source IDs, and SHA-256. `results/task1_finetuning/data_metadata.json`
contains the metadata from the completed preparation job; raw training data is stored separately.

All five Task 1 methods use `configs/task1_finetuning/shared.json`. Full FT and
L2-SP have FP32 parameters; adapters have FP32 trainable factors on a frozen FP16
backbone. OPLoRA bases are computed from the original FP32 starting weights and
remain FP32. Loss is normalized over all causal response tokens in the effective
batch. Recovery includes optimizer, deterministic schedule position, scaler,
RNG, and data position. Extra chunk-boundary checkpoints are recovery-only and
do not add evaluation conditions. Skipped AMP updates are logged explicitly.

Only complete-split evaluations count as assignment results. Limited evaluations
are smoke checks and are labeled as such. GSM8K uses the non-thinking chat template;
likelihood benchmarks use stock completion prompts. No external judge API is used.
Saved experiment metrics are under `results/`.

## Code and results

| Task | Implementation | Configuration and results |
| --- | --- | --- |
| 1. Fine-tuning and retention | [Training, evaluation, and diagnostics](task1_finetuning/) | [Shared training configuration](configs/task1_finetuning/shared.json), [benchmark comparison](results/task1_finetuning/kaggle_benchmark_comparison_full_recovery.csv), [results](results/task1_finetuning/) |
| 2. Visual grounding | [Implementation and data preparation](task2_vlm/IMPLEMENTATION.md) | [Training configuration](configs/task2_vlm/main.json), [split IDs and results](results/task2_vlm/) |
| 3. Test-time compute | [Reproduction commands and scoring protocol](task3_test_time/README.md) | [Offline and live results](results/task3_test_time/) |

The Task 1 notebooks capture the training and evaluation setup. The baseline,
Full FT midpoint at update 625, and all five final checkpoints at update 1,250
use complete evaluation splits. The Task 2 implementation guide documents input
preparation, the curriculum, and the four image conditions. For Task 3, offline
analysis uses the supplied rollouts, with stopping settings selected on development
data before held-out evaluation.

## Sources and attribution

- Assignment manual, model/dataset identities, hyperparameters and protocols.
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B), Transformers and PyTorch APIs.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
  pinned to 0.4.8 for stock benchmark definitions and answer filters.
- [Qualitative prompts](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml)
  from Emergent Misalignment, used as directed by the manual. The exact downloaded
  commit and hash are in the adjacent provenance JSON. Its judging configuration
  is not executed; the assignment calls for greedy responses and manual annotation.
