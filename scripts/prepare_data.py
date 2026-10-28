from common.modal_config import app, data_image, data_volume

prepare_image = data_image.add_local_python_source(
    "task1_finetuning"
)


@app.function(
    image=prepare_image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/data": data_volume},
)
def prepare():
    import json

    from task1_finetuning.prepare_data import prepare_data

    metadata = prepare_data()
    data_volume.commit()

    print(json.dumps(metadata, indent=2))


@app.local_entrypoint()
def main():
    prepare.remote()