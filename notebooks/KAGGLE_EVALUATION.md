# Full LoRA4 benchmark evaluation

Training is complete. Import `pa3_kaggle_lora4_evaluate1250.ipynb` into the
existing private Kaggle notebook. Attach the original private Task1 dataset and
`C:\atml\pa3\outputs\kaggle\pa3-lora4-final-eval-input.zip` as a private input.
The 37,620,123-byte input SHA256 is
`d14a0f0003ec8b434ad042fa23449c59ca2576a250e8f866049059afa22f2636`.

After confirming the prior job is terminal and the draft session is off, set
`BATCH_HANDOFF_CONFIRMED = True`. Check free GPU availability and Internet, then
choose **Save Version → Save & Run All** once. This runs full HellaSwag, BoolQ,
ARC-Easy and GSM8K evaluation with the prescribed protocol, without retraining.
The controller records partial completion explicitly if its time bound is hit.
Download the result archive from the successful saved version for verification.

Checkpoint preflight runs before package installation. It accepts raw ZIPs or
extracted copies only when all four pinned LoRA4 files match; unrelated models'
step-1250 checkpoints are ignored. Changed/missing LoRA4 files fail closed.
The actual LoRA4 input, a duplicate extracted copy and the unrelated LoRA16 final
archive passed this preflight locally. All five code cells compile. GPU benchmark
execution and scores remain pending.

# L2-SP memory preparation

`--method l2sp --memory-strategy cpu-offload` keeps the fixed initial reference,
FP32 AdamW host parameters, gradients and moments in CPU RAM while model parameters,
AMP unscaling and gradient clipping remain on the GPU. It preserves the objective,
hyperparameters and ordinary checkpoint state schema. The default device strategy
is unchanged. CPU tests verify identical updates, skipped nonfinite gradients and
resumption in the real tiny-model training loop.

This is not yet a measured GPU memory-fit or throughput result. Before a full
Kaggle L2-SP run, validate GPU peak memory and host RAM and plan disk capacity for
four full-model checkpoints plus recovery states. Duplicating every full checkpoint
in a final ZIP can exceed available disk; do not launch an unvalidated archive plan.
All future training runs target the full prescribed endpoint, rather than arbitrary
150-update chunks. Monitoring remains off at the user's request.
