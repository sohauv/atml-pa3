"""Remote workers. Launch through scripts.experiments, never directly."""
import time

from common.modal_config import app, data_image, data_volume, checkpoint_volume

worker_image = (data_image
                .pip_install("numpy==2.2.5", "lm_eval==0.4.8", "peft==0.15.2",
                             "matplotlib==3.10.1", "PyYAML==6.0.2", "Pillow==11.1.0",
                             "math-verify[antlr4_13_2]==0.8.0")
                .add_local_python_source("common", "task1_finetuning", "task2_vlm", "task3_test_time")
                .add_local_dir("configs", remote_path="/opt/pa3/configs", ignore=["spending.approved.json"]))


@app.function(image=worker_image, gpu="A100-40GB", cpu=(4, 4), memory=(32768, 32768),
              timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/data": data_volume, "/checkpoints": checkpoint_volume})
def execute(request):
    worker_started = time.time()
    from pathlib import Path
    import importlib.metadata
    from common.io import write_json
    ticket = request["ticket"]
    remaining = min(ticket["seconds"], ticket["deadline"] - time.time())
    if remaining < 180:
        raise TimeoutError("Reservation deadline expired; no new work started")
    data_volume.reload()
    checkpoint_volume.reload()
    run = Path("/checkpoints/runs") / request["run_id"]
    # Ticket metadata is additive; old checkpoints are never removed.
    tickets = Path("/checkpoints/management") / ticket["id"]
    tickets.mkdir(parents=True, exist_ok=True)
    write_json(tickets / "request.json", request)
    write_json(tickets / "environment.json", {
        item.metadata["Name"]: item.version for item in importlib.metadata.distributions()
        if item.metadata.get("Name")})
    checkpoint_volume.commit()
    job = request["job"]
    if ticket.get("gpu"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU was reserved but CUDA is unavailable")
        print(f"[PA3] job={job} requested_gpu={ticket['gpu']} actual_gpu={torch.cuda.get_device_name(0)}", flush=True)
    # Setup/imports/volume commits consume the same provider timeout as work.
    # Recalculate here and retain shutdown time in addition to each job's own
    # checkpoint margin, rather than passing a stale pre-setup duration.
    from common.runtime import worker_budget
    remaining = worker_budget(ticket, worker_started, time.time())
    if remaining < 180:
        raise TimeoutError("Insufficient reservation time after setup; no work started")
    if job == "prepare1":
        from task1_finetuning.prepare_data import prepare_data
        result = prepare_data()
        data_volume.commit()
    elif job == "train1":
        from task1_finetuning.train import train
        result = train(request["method"], run, request["config"],
                       resume=request.get("resume"), max_seconds=remaining,
                       max_updates=request.get("max_updates"), commit=checkpoint_volume.commit)
    elif job in ("cache2", "train2", "evaluate2"):
        from common.io import read_json
        revisions = read_json(request["model_revisions"])
        if job == "cache2":
            from task2_vlm.model import cache_features
            result = cache_features(request["manifest"], request["image_root"], request["features"],
                                    revisions["google/siglip-base-patch16-224"],
                                    max_seconds=remaining, commit=data_volume.commit)
            data_volume.commit()
        elif job == "train2":
            from task2_vlm.train import train
            result = train(request["config"], request["manifest"], request["features"],
                           revisions["Qwen/Qwen3-0.6B"], run, resume=request.get("resume"),
                           max_seconds=remaining, max_updates=request.get("max_updates"),
                           commit=checkpoint_volume.commit)
        else:
            from task2_vlm.train import evaluate
            if not request.get("checkpoint"):
                raise ValueError("Task2 evaluation requires the completed Stage3 checkpoint")
            result = evaluate(request["config"], request["manifest"], request["features"],
                              revisions["Qwen/Qwen3-0.6B"], request["checkpoint"], run,
                              max_seconds=remaining, commit=checkpoint_volume.commit)
    elif job == "live3":
        from task3_test_time.live import run_live
        result = run_live(request["data_root"], request["split_path"], run,
                          max_seconds=remaining, commit=checkpoint_volume.commit)
    else:
        from task1_finetuning.evaluate import evaluate_job
        result = evaluate_job(request, run, remaining, commit=checkpoint_volume.commit)
        checkpoint_volume.commit()
    write_json(tickets / "result.json", result)
    checkpoint_volume.commit()
    return result
