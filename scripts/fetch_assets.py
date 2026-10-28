"""Fetch public assignment inputs; save provenance and never overwrite files."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from common.io import write_json


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ATML-PA3-reproducibility"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="configs/task1_finetuning/first_plot_questions.yaml")
    parser.add_argument("--revision", help="Exact source commit, useful when anonymous GitHub API quota is exhausted")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    repo = "emergent-misalignment/emergent-misalignment"
    revision = args.revision or json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/main"))["sha"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("An exact 40-character Git commit is required")
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/evaluation/first_plot_questions.yaml"
    content = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    write_json(path.with_suffix(".provenance.json"), {
        "source": url, "revision": revision, "sha256": hashlib.sha256(content).hexdigest(),
        "usage": "Only prompts/system fields; no external judge calls."})
    print(path)


if __name__ == "__main__":
    main()
