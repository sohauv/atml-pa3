# Task 3: supplied rollouts and bounded live generation

The PA3 manual is authoritative. This implementation does not regenerate the
supplied pools and does not require images for the offline experiments.

## Public inputs

`download.py` retrieves the five public Parquet exports linked from the manual
and the pinned MathVista `testmini.json`. The original export README is stored
under ignored `data/task3/README.MD`. Downloads never replace existing files.
An interrupted `.part` file is preserved; select a fresh data directory when
retrying rather than deleting an existing artifact automatically.

```powershell
python -m task3_test_time.download --root data/task3 --mathvista-revision c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb
python -m task3_test_time.offline --dataset math --data data/task3 --output outputs/task3/math-v1
python -m task3_test_time.offline --dataset mathvista --data data/task3 --output outputs/task3/mathvista-v1
python -m task3_test_time.figures outputs/task3/math-v1
python -m task3_test_time.figures outputs/task3/mathvista-v1
```

Use a fresh output directory for each offline run. Install the pinned scorer
`math-verify[antlr4_13_2]==0.8.0`, NumPy, pandas, PyArrow, and matplotlib. Model
generation additionally requires the project's pinned Torch/Transformers stack.

## Methodology

- Verify exact one-to-one join keys, complete 32-rollout pools, list lengths,
  and attention prediction positions. Rename the attention table's list-valued
  `n_generated_text_tokens` before joining the scalar generation length.
- Sort string IDs lexicographically and permute with NumPy seed 6304 separately
  for each dataset. Save the 200 development IDs, 800 held-out IDs, first 50
  held-out live IDs, and ten reusable rollout orders per question.
- Form answer groups without reading gold answers. Use Math-Verify on the
  supplied extracted mathematical expressions. Equivalence accepts either
  representation direction and uses the same rule when grading. Symbolic work
  runs in a persistent isolated process with a fixed 10-second comparison limit,
  avoiding Math-Verify's broken nested multiprocessing timeout on Windows.
  Timed-out/failed comparisons return false and are explicitly saved in
  `scorer_diagnostics.json`; no candidate or question is dropped. No fallback
  numeric extractor is added. Identical nonmissing strings are equal.
- MathVista normalization follows the released choice edit-distance rule and
  integer/precision/list conversions, with option letters as canonical labels.
  Null/empty extraction is always one incorrect `NO_ANSWER` group.
- Compute candidate-relative calibration anew for every prefix. Selectors receive
  only candidate answers and traces, never gold answers. Average over orders
  within question before question-level summaries and bootstrap intervals.
- Tune stopping thresholds using development results only, save the selected
  policies before held-out summarization, and use the required fixed-32 fallback.
- Quartile score ties are resolved by stable source rollout order. Undefined
  correlations from constant traces are saved as null, with valid question counts
  in aggregate summaries. They do not remove those questions from accuracy/cost.
- Illustrative token trajectories use the first three saved held-out IDs and
  source rollout IDs 0 and 1, fixed before inspecting performance.

`per_order.jsonl.gz` preserves candidate IDs, predictions, costs, coverage,
and failure decomposition. Per-question CSVs, policy grids, scorer/data
provenance, paired intervals, signals, and figure source tables are saved beside
the aggregate CSVs. Large/raw outputs belong in ignored `outputs/`, not Git.

## Live generation

`live.run_live(data_root, split_path, output, max_seconds=..., commit=...)` is the
Modal integration API. It requires an existing offline `splits.json`, loads the
original model revision from the supplied README in FP16, and uses batch size 1.
The budget-aware project launcher must authorize any remote execution.

Each question saves three independent 1,024-token calls, one 3,072-token call,
and a 2,048-token critique/revision reusing the first parallel response. Calls use
independent saved seeds; the shared initial response is charged to both strategy
totals. Output includes actual prompt/output tokens, wall inference time excluding
model loading, GPU/memory, extraction and cap flags, wrong-to-correct and
correct-to-wrong transitions. Output-token counts include any generated EOS.

Every call is saved and committed immediately. Resuming with the same output
directory reuses completed calls after checking their prompts/seeds/caps; it
does not regenerate successful calls. Partial summaries explicitly identify
fewer than 50 completed questions. They must not be reported as a full experiment.

## Attribution

Answer normalization is an independent implementation of the rules in
[MathVista calculate_score.py](https://github.com/lupantech/MathVista/blob/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb/evaluation/calculate_score.py).
Mathematical equivalence uses [Math-Verify](https://github.com/huggingface/Math-Verify).
Selection, VisConf, consensus, splits, and stopping implement the PA3 equations.
No external LLM judging/API calls are used.
