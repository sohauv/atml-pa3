"""Submit a fully prefunded detached queue; status checks launch no compute."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import uuid

from common.budget import ceiling_cost, reserve_queue
from common.experiment_queue import TERMINAL
from common.io import read_json, write_json


def plan_queue(plan, approval, ledger, queue_id, existing_call_id, now=None):
    now = time.time() if now is None else now
    if ledger.get("active_queue"):
        raise PermissionError("Reconcile the active queue before submitting another")
    controller_seconds = int(plan["controller_seconds"])
    if not 600 <= controller_seconds <= 86400:
        raise ValueError("Invalid controller duration")
    expires = datetime.fromisoformat(approval["expires_at"].replace("Z", "+00:00")).timestamp()
    if now + controller_seconds > expires:
        raise PermissionError("Controller cannot outlive the spending authorization")
    remaining = float(approval["limit_usd"]) - sum(j["reserved_usd"] for j in ledger.get("jobs", []))
    resources = [{"gpu": None, "seconds": controller_seconds, "cpu": .125, "memory_gib": .5,
                  "purpose": "detached queue CPU controller"}]
    reserved = ceiling_cost(None, controller_seconds, .125, .5)
    steps, unfunded = [], []
    for spec in plan["steps"]:
        if spec["job"] not in {"probe1", "train1", "evaluate1", "diagnostics1", "qualitative1", "live3", "cache2", "train2", "evaluate2"}:
            raise ValueError("Unknown queue job")
        seconds = int(spec["seconds"])
        if not 300 <= seconds <= 7200 or spec["gpu"] not in approval["allowed_gpus"]:
            raise ValueError("Queue step resources are invalid or unauthorized")
        cost = ceiling_cost(spec["gpu"], seconds)
        if unfunded or reserved + cost > remaining:
            unfunded.append({"name": spec["name"], "reservation_usd": cost})
            continue
        reserved += cost
        resources.append({"gpu": spec["gpu"], "seconds": seconds, "cpu": 4, "memory_gib": 32,
                          "purpose": spec["name"]})
        config = read_json("configs/task2_vlm/main.json" if spec["job"] in {"cache2", "train2", "evaluate2"}
                           else "configs/task1_finetuning/shared.json")
        payload = {"job": spec["job"], "method": spec.get("method", "full"), "run_id": spec["run_id"].format(short=queue_id[:10]),
                   "config": config, "resume": spec.get("resume"), "checkpoint": spec.get("checkpoint"),
                   "max_updates": spec.get("max_updates"), "probes_dir": spec.get("probes_dir"),
                   "eval_tasks": spec.get("eval_tasks", ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]),
                   "eval_limit": None, "eval_likelihood_batch_size": spec.get("eval_likelihood_batch_size", 8),
                   "data_root": spec.get("data_root", "/data/task3"),
                   "split_path": spec.get("split_path", "/data/task3/analysis/math/splits.json"),
                   "manifest": spec.get("manifest", "/data/task2/prepared/manifest.json"),
                   "features": spec.get("features", "/data/task2/features196"),
                   "image_root": spec.get("image_root", "/data/task2/images"),
                   "model_revisions": spec.get("model_revisions", "/data/task2/model_revisions.json")}
        if "/" in payload["run_id"] or ".." in payload["run_id"]:
            raise ValueError("Unsafe run ID")
        steps.append({"name": spec["name"], "request": payload})
    if not steps:
        raise PermissionError("Remaining allowance cannot fund even the first GPU step; no controller launched")
    request = {"queue_id": queue_id, "existing_call_id": existing_call_id,
               "deadline": now + controller_seconds, "steps": steps, "unfunded_steps": unfunded}
    return request, resources


def sync_status(queue_id, ledger_path, state):
    """Only release queue ownership after a verified terminal remote state."""
    # A failed/ambiguous control call might have left a child running. Retain
    # ownership until its known call IDs are manually verified/reconciled.
    if state["queue_id"] != queue_id or state["phase"] not in {"completed", "budget_limited"}:
        return False
    ledger_path = Path(ledger_path)
    lock = ledger_path.with_suffix(".lock")
    handle = lock.open("x", encoding="utf-8")
    try:
        with handle:
            ledger = read_json(ledger_path)
            if ledger.get("active_queue") != queue_id:
                raise ValueError("Queue ownership mismatch")
            completed = {s["ticket_id"] for s in state["steps"]}
            for ticket in ledger["jobs"]:
                if ticket.get("queue_id") == queue_id:
                    ticket["queue_phase"] = state["phase"]
                    if ticket["id"] in completed:
                        ticket["status"] = "success_queue_ceiling_retained"
            # Ambiguous/failed reservations stay fully charged to the guard.
            ledger.pop("active_queue")
            write_json(ledger_path, ledger)
            return True
    finally:
        lock.unlink(missing_ok=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "submit"):
        p = sub.add_parser(command)
        p.add_argument("--plan", default="configs/experiment_queue.json")
        p.add_argument("--approval", default="configs/spending.approved.json")
        p.add_argument("--ledger", default="runs/budget.json")
        p.add_argument("--wait-call", required=True)
    status = sub.add_parser("status")
    status.add_argument("queue_id")
    status.add_argument("--ledger", default="runs/budget.json")
    status.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        import modal
        path = f"management/queues/{args.queue_id}/status.json"
        state = json.loads(b"".join(modal.Volume.from_name("atml-pa3-checkpoints").read_file(path)))
        write_json(Path("runs/queues") / args.queue_id / "status.json", state)
        if args.sync:
            sync_status(args.queue_id, args.ledger, state)
        print(json.dumps(state, indent=2))
        return
    approval, ledger = read_json(args.approval), read_json(args.ledger)
    queue_id = uuid.uuid4().hex
    request, resources = plan_queue(read_json(args.plan), approval, ledger, queue_id, args.wait_call)
    if args.command == "plan":
        print(json.dumps({"funded_steps": [s["name"] for s in request["steps"]],
                          "unfunded_steps": request["unfunded_steps"],
                          "reservation_usd": sum(ceiling_cost(r["gpu"], r["seconds"], r["cpu"], r["memory_gib"]) for r in resources)}, indent=2))
        return
    import modal
    from scripts.modal_queue import app, orchestrate
    tickets = reserve_queue(args.approval, args.ledger, queue_id, resources)
    for step, ticket in zip(request["steps"], tickets[1:]):
        step["ticket"] = ticket
    local = Path("runs/queues") / queue_id
    write_json(local / "request.json", request)
    # detach=True lets the controller and children outlive this local client,
    # the Codex turn, and Codex usage limits. No client polling loop is required.
    with modal.enable_output(), app.run(detach=True):
        call = orchestrate.with_options(timeout=resources[0]["seconds"]).spawn(request)
        write_json(local / "call.json", {"function_call_id": call.object_id})
    print(json.dumps({"queue_id": queue_id, "controller_call_id": call.object_id,
                      "funded_steps": [s["name"] for s in request["steps"]],
                      "unfunded_steps": request["unfunded_steps"]}, indent=2))


if __name__ == "__main__":
    main()
