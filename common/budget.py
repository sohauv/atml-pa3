"""Conservative local reservation ledger. Creating approval is a human decision.

Reserve the entire job ceiling BEFORE a remote call. Failed/ambiguous calls keep
their reservation. Successful calls may reconcile against the entire local app
lifetime (including image startup), with the same overhead and safety margin.
This is an application guard, not a replacement for a Modal workspace budget.
"""
from datetime import datetime, timezone
import math
from pathlib import Path
import uuid

from common.io import read_json, write_json

# USD/second from https://modal.com/pricing, checked 2026-10-28.
GPU_RATES = {"T4": .000164, "L4": .000222, "A10G": .000306,
             "A100-40GB": .000583, "A100-80GB": .000694,
             "L40S": .000542, "H100": .001097}
CPU_RATE = .0000131
MEMORY_RATE = .00000222


def ceiling_cost(gpu, seconds, cpu=4, memory_gib=32):
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("A finite positive job ceiling is required")
    if cpu <= 0 or memory_gib <= 0 or not all(map(math.isfinite, (cpu, memory_gib))):
        raise ValueError("Resource ceilings must be finite and positive")
    # Full requested resources plus 25% contingency and two minutes for teardown.
    rate = GPU_RATES[gpu] if gpu else 0
    return (seconds + 120) * (rate + cpu * CPU_RATE + memory_gib * MEMORY_RATE) * 1.25


def reserve(approval_path, ledger_path, *, gpu, seconds, purpose, cpu=4, memory_gib=32):
    approval = read_json(approval_path)
    if approval.get("approved") is not True or not approval.get("approved_by"):
        raise PermissionError("No human-approved spending limit")
    now = datetime.now(timezone.utc)
    expires = datetime.fromisoformat(approval["expires_at"].replace("Z", "+00:00"))
    if now >= expires:
        raise PermissionError("Spending authorization has expired")
    if gpu and gpu not in approval.get("allowed_gpus", []):
        raise PermissionError("GPU is outside the authorization")
    cap = float(approval["limit_usd"])
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("Invalid approved limit")
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock = ledger_path.with_suffix(".lock")
    # Exclusive file creation makes simultaneous local launches fail closed.
    handle = lock.open("x", encoding="utf-8")
    try:
        with handle:
            ledger = read_json(ledger_path) if ledger_path.exists() else {"jobs": []}
            if ledger.get("authorization_id", approval["authorization_id"]) != approval["authorization_id"]:
                raise PermissionError("Use a separate ledger for each authorization")
            charge = ceiling_cost(gpu, seconds, cpu, memory_gib)
            spent = sum(job["reserved_usd"] for job in ledger["jobs"])
            if spent + charge > cap:
                raise PermissionError(f"Budget exhausted: ${spent:.2f} reserved; ${charge:.2f} needed; cap ${cap:.2f}")
            job = {"id": uuid.uuid4().hex, "purpose": purpose, "gpu": gpu,
                   "seconds": seconds, "reserved_usd": charge,
                   "reserved_at": now.isoformat(), "status": "reserved"}
            ledger["authorization_id"] = approval["authorization_id"]
            ledger["jobs"].append(job)
            write_json(ledger_path, ledger)
            return job
    finally:
        # Close first: Windows cannot unlink an open file.
        lock.unlink(missing_ok=True)


def settle_success(ledger_path, ticket_id, elapsed_seconds, *, cpu=4, memory_gib=32):
    """Retain a padded measured ceiling only after confirmed successful completion.

    A single local app lifetime covers all serial infrastructure retries. Never
    settle failed/unknown calls or provider bills we have not actually inspected.
    """
    ledger_path = Path(ledger_path)
    lock = ledger_path.with_suffix(".lock")
    handle = lock.open("x", encoding="utf-8")
    try:
        with handle:
            ledger = read_json(ledger_path)
            jobs = [job for job in ledger["jobs"] if job["id"] == ticket_id]
            if len(jobs) != 1:
                raise ValueError("Unknown or ambiguous reservation")
            job = jobs[0]
            if job["status"] != "reserved":
                raise ValueError("Reservation already settled")
            measured = ceiling_cost(job["gpu"], elapsed_seconds, cpu, memory_gib)
            job["elapsed_app_seconds"] = elapsed_seconds
            job["original_reserved_usd"] = job["reserved_usd"]
            # Include unexpectedly long image startup rather than understating it.
            job["reserved_usd"] = measured
            job["status"] = "success_conservative_estimate"
            write_json(ledger_path, ledger)
            return job
    finally:
        lock.unlink(missing_ok=True)
