"""Single budget-gated entrypoint for remote work. No compute on --help/plan.

Run as `python -m scripts.experiments plan` or `... run --help`.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import modal
import sys

from common.budget import ceiling_cost, reserve, settle_success
from common.io import append_jsonl, read_json, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--gpu", default="A100-40GB")
    plan.add_argument("--seconds", type=int, default=1800)
    run = sub.add_parser("run")
    run.add_argument("--job", choices=["prepare1", "train1", "evaluate1", "diagnostics1", "probe1", "qualitative1", "cache2", "train2", "evaluate2", "live3"], required=True)
    run.add_argument("--method", choices=["full", "l2sp", "lora4", "lora16", "oplora16"], default="full")
    run.add_argument("--gpu", choices=["L4", "A100-40GB", "A100-80GB", "L40S"], default="A100-40GB")
    run.add_argument("--seconds", type=int, default=1800)
    run.add_argument("--approval", default="configs/spending.approved.json")
    run.add_argument("--ledger", default="runs/budget.json")
    run.add_argument("--run-id", default=None)
    run.add_argument("--resume")
    run.add_argument("--checkpoint")
    run.add_argument("--probes-dir")
    run.add_argument("--eval-tasks", default="gsm8k_cot,hellaswag,boolq,arc_easy")
    run.add_argument("--eval-limit", type=int, help="Smoke validation only; never use limited scores in assignment results")
    run.add_argument("--manifest", default="/data/task2/prepared/manifest.json")
    run.add_argument("--features", default="/data/task2/features196")
    run.add_argument("--image-root", default="/data/task2/images")
    run.add_argument("--model-revisions", default="/data/task2/model_revisions.json")
    run.add_argument("--data-root", default="/data/task3")
    run.add_argument("--split-path", default="/data/task3/analysis/math/splits.json")
    run.add_argument("--max-updates", type=int)
    run.add_argument("--config", help="Defaults to the prescribed Task1/Task2 configuration for the selected job")
    return p


def main():
    # Rich/Modal emits Unicode status markers; Windows redirected stdout otherwise
    # defaults to cp1252 and can fail before a remote call is submitted.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args()
    if args.command == "plan":
        print(json.dumps({"gpu": args.gpu, "seconds": args.seconds,
                          "reservation_usd": ceiling_cost(args.gpu, args.seconds)}, indent=2))
        return
    if not 300 <= args.seconds <= 7200:
        raise ValueError("Use job ceilings between 300 and 7200 seconds")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("max-updates must be positive")
    if args.job in ("train1", "train2") and args.resume and not args.run_id:
        raise ValueError("Resume requires the original run-id")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid run ID")
    default_config = "configs/task2_vlm/main.json" if args.job in ("cache2", "train2", "evaluate2") else "configs/task1_finetuning/shared.json"
    config = read_json(args.config or default_config)
    # Import before reserving, but app.run() only starts AFTER reservation succeeds.
    if args.job == "prepare1":
        from scripts.modal_prepare_job import app, prepare_execute as execute
    else:
        from scripts.modal_jobs import app, execute
    gpu = None if args.job == "prepare1" else args.gpu
    ticket = reserve(args.approval, args.ledger, gpu=gpu, seconds=args.seconds,
                     purpose=f"{args.job}/{args.method}/{run_id}")
    # Absolute deadline survives automatic infrastructure retries/preemption.
    ticket["deadline"] = time.time() + args.seconds
    request = {"job": args.job, "method": args.method, "run_id": run_id,
               "resume": args.resume, "checkpoint": args.checkpoint,
               "probes_dir": args.probes_dir,
               "eval_tasks": args.eval_tasks.split(","), "eval_limit": args.eval_limit,
               "manifest": args.manifest, "features": args.features, "image_root": args.image_root,
               "model_revisions": args.model_revisions, "data_root": args.data_root,
               "split_path": args.split_path,
               "max_updates": args.max_updates, "config": config, "ticket": ticket}
    local = Path("runs") / ticket["id"]
    local.mkdir(parents=True, exist_ok=False)
    write_json(local / "request.json", request)
    app_started = time.monotonic()
    try:
        with modal.enable_output(), app.run():
            function = execute.with_options(gpu=gpu, timeout=args.seconds,
                                            cpu=(4, 4), memory=(32768, 32768))
            call = function.spawn(request)
            write_json(local / "call.json", {"function_call_id": call.object_id})
            try:
                # Modal execution timeouts exclude scheduling; bound the local
                # queue wait too so scarce GPU capacity cannot stall management.
                result = call.get(timeout=args.seconds + 120)
            except TimeoutError:
                call.cancel(terminate_containers=True)
                raise TimeoutError("Job or GPU queue exceeded the reservation window; prior checkpoints retained")
        write_json(local / "result.json", result)
        settle_success(args.ledger, ticket["id"], time.monotonic() - app_started)
        print(json.dumps(result, indent=2))
    except BaseException as error:
        write_json(local / "failure.json", {"error": repr(error), "reservation_retained": True})
        raise


if __name__ == "__main__":
    main()
