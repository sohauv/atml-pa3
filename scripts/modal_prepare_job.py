import time
from common.modal_config import app, data_image, data_volume

prepare_image = data_image.add_local_python_source("common", "task1_finetuning")


@app.function(image=prepare_image, cpu=(4, 4), memory=(32768, 32768),
              timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/data": data_volume})
def prepare_execute(request):
    if request["ticket"]["deadline"] - time.time() < 120:
        raise TimeoutError("Preparation reservation expired")
    from task1_finetuning.prepare_data import prepare_data
    data_volume.reload()
    result = prepare_data()
    data_volume.commit()
    return result
