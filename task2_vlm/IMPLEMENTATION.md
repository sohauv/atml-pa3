Task 2 implementation follows manual pages 7-10. No real training or evaluation
results are bundled with this source.

Prepare official metadata locally with `python -m task2_vlm download-metadata`,
then `python -m task2_vlm prepare`, then `python -m task2_vlm download-images`.
Metadata downloads include the original COCO instance/caption annotations and
VQAv2 question, answer, and complementary-pair files; only selected images are
downloaded. Raw files and images belong under the ignored `data/` directory.
Download artifacts are preserved and never overwritten or deleted.
The custom COCO hostname currently fails TLS hostname validation; downloads use
certificate-valid HTTPS endpoints for the same official S3 bucket. Large archives
transfer as verified byte ranges with ETag/source identity checks. Selected image
transfers use bounded parallelism and Pillow validation.

`data.prepare_data(paths, output_dir, config)` writes a manifest with all selected
image/question/pair/caption IDs and source hashes. Natural VQA quotas are exactly
640/256/640 training, 12/8/12 development, and 36/24/36 test pairs. All evaluation
suites are mutually image-disjoint, and isolated from training. Grounding
evaluation uses an explicitly configured equal presence/counting split (the
manual fixes total pair counts but does not prescribe that split). Grounding
labels reflect annotation counts, including crowd annotations; absence means
the category is unannotated, and annotation ambiguity remains a limitation.
Preparation uses `ijson==3.4.0.post0` to stream metadata and discard segmentation.
When `data/task2/preflight_natural.json` exists, preparation rechecks its original
source checksums, pair membership, consensus, and quotas, then streams selected
QAs only. Exact RNG consumption is replayed before grounding/caption selection;
a dedicated test proves matching selected IDs and subsequent RNG state. Caption
strings are retained only after selecting image IDs.
`python -m task2_vlm export-ids` publishes small CSVs containing only selected
image, caption, question, pair, and category IDs plus counts/source checksums
under `results/task2_vlm/selection_ids/`. Raw text, answers, images, features,
and datasets remain in the ignored `data/` tree.

Pin model revisions once with `python -m task2_vlm pin-models`. Pass those pinned
revisions to GPU functions. `model.cache_features(manifest_path, image_root,
output_dir, vision_revision, max_seconds=..., commit=...)` caches [196,768]
FP16 feature tensors. Transformers 4.51.3 already applies SigLIP post_layernorm
to returned last_hidden_state; this code does not apply it twice. Deterministic
processor configuration is pinned with the model revision. The encoder is freed
and CUDA cache emptied after caching.

`train.train(config, manifest_path, feature_dir, model_revision, run_dir,
resume=None, max_seconds=1800, max_updates=None, commit=...)` executes bounded
training chunks. Use a fresh run directory initially; pass the original run
directory and returned complete checkpoint to resume. Connector calibration
uses the first 32 selected training captions' image features and question text,
matching their mean token L2 norms; the measured scale is recorded. This is a
training-only rule fixed before any development results.

The caption stage makes two exact passes over 18,000 examples. Later stages draw
eight independent two-example units per effective batch using the documented
probabilities. Sampler state is serialized, and no mixture quota is enforced.
All trainable parameters, including biases, receive AdamW weight decay 0.01;
betas=(0.9,0.999) and epsilon=1e-8 are standard optimizer defaults because the
Task2 manual does not specify them. Warmup rounds 5% of each stage upward to a
whole update. Optimizer moments continue between stages; adapter parameters
are added when Stage2 begins. Training losses weight examples equally.

Resumable checkpoints include connector/adapter states, optimizer, deterministic
scheduler position, scaler, full RNG state, sampler state, stage progress, and
configuration/provenance identity. Required 250-update and stage-end checkpoints
are supplemented by recovery snapshots at bounded chunk exits. No checkpoint is
overwritten. Interrupted named checkpoints receive unique indexed retry paths,
preserving the partial originals. The calibrated untrained state is saved before
the first development sweep. Development likelihood is evaluated at update0, every500 updates,
and all stage endpoints. Completed development records and checkpoints are
committed through the caller-provided Modal volume callback.
Development sweeps check an absolute deadline per microbatch and reserve time
for serialization. Missing required sweeps are retried on resume, with attempt
IDs distinguishing partial raw records from the completed sweep.

`train.evaluate(config, manifest_path, feature_dir, model_revision, checkpoint,
output_dir, max_seconds=..., commit=...)` enforces the final4500-update checkpoint,
then runs batch8 greedy generation with a16-token cap under all four image
conditions. Original references remain fixed. Answer NLL excludes termination;
training and caption-development loss include it. No-image sequences remove the
visual tokens and rebuild masks/position IDs. Saved mismatched mappings exclude
both original pair images. Metrics bootstrap complete pairs with4000 resamples.
Final evaluation reuses completed conditions only under identical checkpoint,
model, data, feature, and configuration identities. Partial attempts remain in
unique raw files. Additive attempt logs record cumulative runtime, failed-call
overhead, and incomplete timing if a hard-killed attempt lacks a final duration.
`figures.make_figures(run_dir, evaluation_dir, output_dir)` creates learning plots
and compact final/resource CSVs from actual saved measurements only.
`figures.make_examples(evaluation_dir, image_root, output_dir)` exports aligned
image pairs using the lexicographically first pair per answer type per suite,
independent of predictions. No report interpretation or analysis is generated.

Sources: VQA normalization is based on the official evaluator by Aishwarya
Agrawal et al. at
https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py .
The score itself implements the manual's explicit leave-one-out equation.
The SigLIP normalization contract was checked against
https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/siglip/modeling_siglip.py .
Dataset URLs and complementary-pair format follow https://visualqa.org/download.html .

Validation: synthetic CPU tests check frozen-backbone connector gradients,
visual insertion/masking/positions, no-image removal, original references under
swaps, mismatch exclusion, split isolation, sampler resume, stage boundaries,
VQA normalization, and leave-one-out soft accuracy. These are not evidence that
real-data training fits a particular GPU or achieves any accuracy. Real-data
downloads, pinned-model integration, training, and final evaluation must be
executed separately under the approved budget.
