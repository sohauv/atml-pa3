"""Download only the assignment's public offline inputs, never model weights."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

from common.io import write_json

FILES = {
    "qwen25-math/generations.parquet": "1SaWxZ4mpycw2BWZ5G0dahoYZ6Hh8tPWP",
    "qwen25-math/probability_metrics.parquet": "1RCYHrTnUKtF7gHLSTe92x6W9ZsCOekfQ",
    "qwen25vl-mathvista/generations.parquet": "1sedMpWLI_UgPvAQVj7tS3HYVH0ARS1yA",
    "qwen25vl-mathvista/probability_metrics.parquet": "1f2XAu_J5NviYHEhtkRrFbKc7b2us1WiW",
    "qwen25vl-mathvista/attention_metrics.parquet": "1Nzg2VwE2qFzA10BF1lTxEcBZCePBe51q",
}


def download(url, path, expected_parquet=False):
    path = Path(path)
    if path.exists():
        if expected_parquet:
            with path.open("rb") as stream:
                if stream.read(4) != b"PAR1":
                    raise ValueError(f"Existing file is not Parquet: {path}")
        return {"source": url, "bytes": path.stat().st_size, "existing": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"Prior partial download preserved at {temporary}; choose another directory")
    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "ATML-PA3-data/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, temporary.open("xb") as stream:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("Drive returned HTML rather than the supplied data; use the public download link")
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
            digest.update(block)
    if expected_parquet:
        with temporary.open("rb") as stream:
            if stream.read(4) != b"PAR1":
                raise ValueError(f"Not a Parquet download: {temporary}")
    temporary.rename(path)
    return {"source": url, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def fetch_all(root="data/task3", mathvista_revision="main"):
    root = Path(root)
    manifest = {}
    for relative, file_id in FILES.items():
        print(f"Downloading {relative}", flush=True)
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        manifest[relative] = download(url, root / relative, True)
    url = f"https://raw.githubusercontent.com/lupantech/MathVista/{mathvista_revision}/data/testmini.json"
    manifest["testmini.json"] = download(url, root / "testmini.json")
    write_json(root / "downloads.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/task3")
    parser.add_argument("--mathvista-revision", required=True)
    args = parser.parse_args()
    fetch_all(args.root, args.mathvista_revision)
