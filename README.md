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

The initial allowance on 28 October was $10. On 29 October the user explicitly
raised PA3 authorization to **$15 total including all earlier jobs**, with a
$15 workspace usage cap and $0 out-of-pocket spend limit. Do not modify
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
Task 2 and Task 3 code is implemented and has CPU fixture/integration tests;
pretrained GPU experiments and complete-split evaluations are tracked separately
in `PROGRESS.md`. A passing CPU test is not a completed experiment.

## Detached experiment queue

`scripts.queue` runs a small CPU controller on Modal with `app.run(detach=True)`.
It waits for the known current GPU call, then advances the fixed plan in
`configs/experiment_queue.json` without a Codex turn or local polling process.
The computer/client can disconnect after submission. GPU workers start only for
their own steps; they do not sit idle between stages. Modal account limits,
preemption, or external-service failures can still stop work.
An older non-detached dependency still needs its original client until that
dependency finishes; the newly queued steps do not have that dependency.

All controller and GPU ceilings are atomically reserved before launch, including
prior experiment reservations. Only the affordable prefix is funded; remaining
steps are reported as unfunded. A queue owns the ledger exclusively, preventing
the chat/monitor from launching overlapping standalone jobs. Ambiguous submission
or failure stops the queue and retains budget/ownership pending reconciliation.
Completed worker outputs and controller state are saved on the checkpoint volume.

```powershell
python -m scripts.queue plan --wait-call <existing-function-call-id>
python -m scripts.queue submit --wait-call <existing-function-call-id>
python -m scripts.queue status <queue-id> --sync
```

The default plan caches baseline probes, resumes the fixed Task3 live comparison,
evaluates the three baseline likelihood benchmarks with batch8, then continues
FullFT only if that final step was affordable. GSM8K generation stays batch1.
No queue retry invents a new starting checkpoint or changes the prescribed seeds.
Funding a later queue requires available already-authorized budget, or explicit
human authorization for a higher total; raising account credits is not assumed
to authorize additional PA3 spend.

## Task 1 execution commands

Preparation reuses a complete checksummed dataset and refuses to replace partial
or inconsistent existing data. All GPU commands require the approved local ledger.

```powershell
python -m scripts.experiments run --job prepare1 --seconds 1800
python -m scripts.experiments run --job train1 --method full --gpu A100-40GB --seconds 6000 --run-id full-<unique-id>
python -m scripts.experiments run --job train1 --method full --gpu A100-40GB --seconds 6000 --run-id full-<same-id> --resume /checkpoints/runs/full-<same-id>/<completed-checkpoint>
python -m scripts.experiments run --job probe1 --gpu A100-40GB --seconds 600 --run-id probes-<unique-id>
python -m scripts.experiments run --job evaluate1 --gpu A100-40GB --seconds 3600 --run-id baseline-<unique-id>
python -m scripts.experiments run --job evaluate1 --gpu A100-40GB --seconds 3600 --run-id full-midpoint-<unique-id> --checkpoint /checkpoints/runs/full-<id>/step-0625
python -m scripts.experiments run --job diagnostics1 --gpu A100-40GB --seconds 1200 --run-id diagnostics-<unique-id> --checkpoint /checkpoints/runs/<id>/<checkpoint> --probes-dir /checkpoints/runs/<probe-id>
python -m scripts.experiments run --job qualitative1 --gpu A100-40GB --seconds 1200 --run-id qualitative-<unique-id> --checkpoint /checkpoints/runs/<id>/<checkpoint>
```

Repeat training independently for `l2sp`, `lora4`, `lora16`, and `oplora16` with
fresh run IDs; never substitute another method's checkpoint as a starting point.
Commands are bounded chunks, not a reduced training recipe. Continue each method
to update1250 when authorized funds permit. Evaluate baseline once, FullFT625,
and all five final1250 checkpoints. `--eval-tasks` selects comma-separated tasks
for smaller recoverable calls; `--eval-limit` is smoke-only and excluded from final
benchmark tables. Cache the64 baseline activation probes before inspecting model
benchmark outputs. Omit `--checkpoint` for unchanged-baseline qualitative responses.

```powershell
python -m scripts.collect --remote /runs/<run-id> --output results/task1_finetuning/<run-id>
.\.venv\Scripts\python.exe -m task1_finetuning.figures --results results/task1_finetuning --output report/figures/task1-<unique-id>
```

Figure exports retain empty cells for missing conditions and label partial runs;
their manifests contain source hashes and completion status. Per-example model
outputs stay in saved evaluation directories, separate from small summary tables.

## Task 2 execution

See `task2_vlm/IMPLEMENTATION.md` for APIs, source attribution and chosen defaults
where the manual leaves a choice. Original-source parsing uses `ijson` to avoid
retaining COCO segmentation and all VQA answer dictionaries simultaneously.

```powershell
.\.venv\Scripts\python.exe -m task2_vlm download-metadata
.\.venv\Scripts\python.exe -m task2_vlm pin-models
.\.venv\Scripts\python.exe -m task2_vlm prepare
.\.venv\Scripts\python.exe -m task2_vlm download-images
python -m scripts.upload_inputs --source data/task2/prepared --remote /task2/prepared-v1
python -m scripts.upload_inputs --source data/task2/images --remote /task2/images-v1
```

Upload the model revision JSON separately to a fresh volume path, then use the
budgeted `cache2`, `train2`, `evaluate2` jobs with explicit `--manifest`,
`--image-root`, `--features`, and `--model-revisions` paths. `cache2` is resumable
and commits feature files; `train2` accepts `--run-id`/`--resume` like Task1. Final
test evaluation rejects checkpoints before completion of all4500 updates.

Before resuming a large partial feature cache, validate its immutable tensors on
CPU rather than downloading them repeatedly while paying for an allocated GPU:

```powershell
python -m scripts.experiments run --job verify_cache2 --seconds 600 --run-id cache-verification-<unique-id> --features /data/task2/features196-v2
```

This reserves CPU/RAM only. It checks actual tensor bytes, shape, dtype and finite
values, then writes a provenance-bound SHA256 index without changing features.
Set `verified_index` in the next cache queue step to the returned `index_path`.
GPU resumes check provenance and file sizes; this relies on the cache writer's
no-overwrite rule. A changed provenance or invalid tensor fails closed. The
verification count may describe a partial cache and never implies training is ready.

## Task 3 execution

See `task3_test_time/README.md` for scorer versions, supplied file IDs, exact
commands and output schemas. Offline analysis needs neither GPUs nor images and
is run locally using `requirements-analysis.txt`. Development policies are frozen
before held-out processing. Incomplete analysis attempts are preserved under
distinct output directories. Symbolic-equivalence timeouts are fixed/versioned
and diagnosed explicitly; questions and missing extractions are never dropped.

The budgeted `live3` job uses only the fixed50 held-out MATH IDs, original pinned
Qwen2.5-3B revision, and required three inference strategies. Upload only the
necessary generations/splits/provenance with `scripts.upload_live_inputs`, then
pass the resulting `--data-root` and `--split-path` to `scripts.experiments`.
Completed per-question live calls are reused on resume; the shared initial answer
is charged to both parallel/refinement strategies. Plot and table tools consume
actual saved results and never fill absent measurements.

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
