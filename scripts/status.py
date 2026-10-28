"""Read-only Modal call/volume inspection; does not launch remote compute."""
import argparse
import json
from pathlib import Path

import modal

from common.io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call-id")
    parser.add_argument("--volume", default="atml-pa3-checkpoints")
    parser.add_argument("--path", default="/")
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--ledger", default="runs/budget.json")
    args = parser.parse_args()
    if Path(args.ledger).exists():
        ledger = read_json(args.ledger)
        print(json.dumps({"reserved_usd": sum(j["reserved_usd"] for j in ledger["jobs"]),
                          "jobs": ledger["jobs"]}, indent=2))
    if args.call_id:
        call = modal.FunctionCall.from_id(args.call_id)
        try:
            print(json.dumps(call.get(timeout=0), indent=2))
        except TimeoutError:
            print("Remote call is still running")
    else:
        volume = modal.Volume.from_name(args.volume)
        if args.read:
            print(b"".join(volume.read_file(args.path)).decode("utf-8"))
        else:
            for entry in volume.iterdir(args.path):
                print(entry.path, entry.size)


if __name__ == "__main__":
    main()
