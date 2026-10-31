# LoRA4 saved background continuation

Use `pa3_kaggle_lora4_resume839.ipynb` for one bounded continuation from the
verified step-839 checkpoint. It keeps the original source commit, packages,
training recipe, optimizer/scaler/RNG state and training-data checksum.

1. Stop the previous draft session only after confirming no training is active.
   The local step-839 backup is already verified.
2. Import the notebook into the existing private Kaggle notebook.
3. Attach the original private Task1 input and upload
   `C:\atml\pa3\outputs\kaggle\pa3-lora4-step839-restore.zip` as a **private**
   dataset. Keep the notebook private. Enable Internet and the free T4 GPU.
4. Set `BATCH_HANDOFF_CONFIRMED = True` after the old draft has stopped.
5. Launch **Save Version > Save & Run All**, rather than running an interactive
   draft. This version starts fresh and restores its checkpoint from attached
   inputs. Do not launch another training session while it runs.
6. When the version finishes, inspect its **Output** files. Download
   `checkpoint-backup-evidence.json` and the named `lora4-backup-step-NNNN.zip`.
   Verify the actual local archive's SHA256, ZIP CRC, recovery files and required
   checkpoint markers before the next continuation.

The notebook requests at most 150 additional updates and 1,800 training seconds;
the time guard can stop earlier. This is one chunk, not completion of all 1,250
updates. The initial guard deliberately prevents an accidental duplicate run.
Missing or changed step-839 input bytes fail before training. Existing run
directories are never overwritten. A printed completion message alone does not
confirm preserved checkpoint files.

Kaggle documentation:
https://www.kaggle.com/docs/notebooks

Local validation: all five code cells compile; CPU checks passed for exact
restoration of all 17 pinned input files, overwrite rejection, corrupted recovery
bytes rejected before copying, and the unconfirmed-handoff guard. GPU execution
of this new notebook has not yet been performed.
