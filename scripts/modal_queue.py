"""Small CPU controller; GPU workers exist only while executing a step."""
import modal

from scripts.modal_jobs import app, execute
from common.modal_config import checkpoint_volume

controller_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "common", "scripts", "task1_finetuning", "task2_vlm", "task3_test_time")


@app.function(image=controller_image, cpu=(.125, .125), memory=(512, 512),
              timeout=10800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/checkpoints": checkpoint_volume})
def orchestrate(request):
    from common.experiment_queue import run_queue
    checkpoint_volume.reload()

    def submit(payload, gpu, seconds):
        return execute.with_options(gpu=gpu, timeout=seconds, cpu=(4, 4), memory=(32768, 32768)).spawn(payload).object_id

    return run_queue(request, "/checkpoints/management/queues/" + request["queue_id"],
                     submit=submit,
                     get=lambda call_id, timeout: modal.FunctionCall.from_id(call_id).get(timeout=timeout),
                     cancel=lambda call_id: modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True),
                     commit=checkpoint_volume.commit)
