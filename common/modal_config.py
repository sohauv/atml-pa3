import modal

app = modal.App("atml-pa3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "accelerate==1.6.0",
    )
    .env({"HF_HOME": "/data/huggingface"})
)

data_image = image.pip_install("datasets==3.5.0")

data_volume = modal.Volume.from_name("atml-pa3-data")
checkpoint_volume = modal.Volume.from_name("atml-pa3-checkpoints")