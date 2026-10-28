"""Deterministic detached orchestration. No LLM or new budget decisions here."""
from pathlib import Path
import time

from common.io import append_jsonl, read_json, write_json

TERMINAL = {"completed", "budget_limited", "failed", "needs_reconciliation", "deadline_reached"}


def run_queue(request, root, *, submit, get, cancel, commit=lambda: None, clock=time.time):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "status.json"
    if (root / "request.json").exists():
        if read_json(root / "request.json") != request:
            raise ValueError("Queue identity changed; use a fresh ID")
    else:
        write_json(root / "request.json", request)
    state = read_json(state_path) if state_path.exists() else {
        "queue_id": request["queue_id"], "phase": "wait_existing", "next_index": 0,
        "steps": [], "started_at": clock(), "existing_call_id": request.get("existing_call_id")}

    def save():
        write_json(state_path, state)
        append_jsonl(root / "events.jsonl", {"at": clock(), "phase": state["phase"], "next_index": state["next_index"]})
        commit()

    def fail(phase, error):
        state.update(phase=phase, error=str(error), ended_at=clock())
        save()
        return state

    if state["phase"] in TERMINAL:
        return state
    if state["phase"] == "submitting":
        return fail("needs_reconciliation", "Interrupted submission is ambiguous; no duplicate worker will be launched")
    deadline = request["deadline"]
    if state["phase"] == "wait_existing":
        save()
        if state.get("existing_call_id"):
            try:
                # A failed dependency also stops the queue. Never assume that a
                # timeout or lost response means the prior GPU has stopped.
                state["existing_result"] = get(state["existing_call_id"], max(1, deadline - clock() - 180))
            except Exception as error:
                return fail("failed", f"Prior job was not confirmed complete: {error!r}")
        state["phase"] = "ready"
        save()
    while state["next_index"] < len(request["steps"]):
        index = state["next_index"]
        step = request["steps"][index]
        if state["phase"] == "ready":
            if clock() + step["ticket"]["seconds"] + 180 >= deadline:
                return fail("deadline_reached", "No new worker started without enough controller time")
            state.update(phase="submitting", current_index=index)
            save()  # write-ahead record makes ambiguous submission fail closed
            payload = {**step["request"], "ticket": {**step["ticket"],
                       "deadline": min(deadline - 120, clock() + step["ticket"]["seconds"])}}
            try:
                state["call_id"] = submit(payload, step["ticket"]["gpu"], step["ticket"]["seconds"])
            except Exception as error:
                return fail("needs_reconciliation", f"Submission may have reached Modal: {error!r}")
            state["phase"] = "waiting_for_step"
            save()
        try:
            result = get(state["call_id"], min(step["ticket"]["seconds"] + 120, max(1, deadline - clock() - 60)))
        except TimeoutError as error:
            cancel(state["call_id"])
            return fail("failed", f"Worker/queue timeout; prior checkpoints preserved: {error!r}")
        except Exception as error:
            return fail("failed", f"Worker failed; no automatic retry: {error!r}")
        state["steps"].append({"index": index, "ticket_id": step["ticket"]["id"],
                               "call_id": state["call_id"], "result": result})
        state["next_index"] += 1
        state["phase"] = "ready"
        state.pop("call_id", None)
        save()
    state.update(phase="budget_limited" if request.get("unfunded_steps") else "completed", ended_at=clock(),
                 unfunded_steps=request.get("unfunded_steps", []))
    save()
    return state
