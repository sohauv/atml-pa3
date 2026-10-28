# ATML PA3

Reproducible experiments for EE-5102/CS-6304 Programming Assignment 3.
The local `ATML-PA3.pdf` is the requirements authority. This repository is under
active implementation; incomplete runs and smoke checks are never final results.

## Environment

Use Python 3.11 on Modal and Python 3.12 for local CPU validation.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The existing Modal volumes are `atml-pa3-data` and `atml-pa3-checkpoints`.
`common/modal_config.py` deliberately requires them to exist. GPU training uses
the pinned CUDA Torch image on Modal; local CPU tests do not reproduce GPU memory
or mixed-precision throughput. Credentials stay in Modal/GitHub's local credential
stores and must never be added to the repository.

## Spending and preservation

All managed remote work goes through `python -m scripts.experiments`. A job must
reserve its entire cost ceiling before submitting work. The local approval and
ledger are ignored by Git. `configs/spending.example.json` is deliberately
unapproved; only populate a real approval following explicit human authorization.

The approved allowance on 28 October 2026 is $10 total for PA3, within the user's
$10 monthly Modal usage limit and $0 out-of-pocket spend limit. Do not modify
workspace billing settings. Credits are not assumed to be an extra allowance.
Reservations include requested CPU/RAM, a two-minute overhead allowance, and a
25% contingency. They are conservative estimates, not provider billing records.
Failed or ambiguous calls keep their reservation. Job startup deadlines survive
infrastructure retries; GPU jobs run sequentially with bounded timeouts.

Existing datasets and checkpoints are preserved. New runs have unique IDs;
completed checkpoints have a `complete.json` marker written last. A partial
checkpoint is never used for recovery. A failed write is preserved for inspection.

```powershell
python -m scripts.experiments plan --gpu A100-40GB --seconds 1200
python -m scripts.status --volume atml-pa3-checkpoints --path /
```

## Methodology

Task 1 uses the first 40,000 eligible MetaMathQA examples in source order, no
shuffle/truncation, seed 6304, Qwen3-0.6B non-thinking templates, and response/EOT
labels. Filtering uses NFKC, casefold, and collapsed whitespace on both query and
original question against GSM8K test questions. Dataset metadata records exact
upstream revisions, source IDs, and SHA-256. `results/task1_finetuning/data_metadata.json`
contains the metadata from the completed preparation job; raw training data stays
on the Modal data volume.

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
Task 2 and Task 3 modules and validation are being completed independently.

## Validation

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests exercise budget refusal/locking, token-weighted accumulation, projected
adapter geometry and merge equivalence, padding, and gold-free candidate selection.
GPU validation and full experiment results are recorded separately from CPU tests.

## Sources and attribution

- Assignment manual, model/dataset identities, hyperparameters and protocols.
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B), Transformers and PyTorch APIs.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
  pinned to 0.4.8 for stock benchmark definitions and answer filters.
- [Qualitative prompts](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml)
  from Emergent Misalignment, used as directed by the manual. The exact downloaded
  commit and hash are in the adjacent provenance JSON. Its judging configuration
  is not executed; the assignment calls for greedy responses and manual annotation.
- [Modal pricing](https://modal.com/pricing), checked 28 October 2026. Rates are
  estimates before taxes and account-specific credits/limits.

Training/adapters/selection implementations directly implement the assignment's
equations; no paper implementation has been copied. Preserve source attribution
when adding external code. The manual permits coding assistance but prohibits
generative AI from writing report language, interpretation or analysis. The PDF
report must be written by the student; machine-readable results and plotting code
support that work.
