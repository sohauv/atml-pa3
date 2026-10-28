"""Download original public COCO/VQAv2 metadata and only selected image files.

No overwrites or deletions. Run metadata() first, data.prepare_data() second, then
images() with the saved manifest. Downloads are local CPU/network operations.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
import urllib.error
import urllib.request
import zipfile

from common.io import read_json, sha256, write_json

ARCHIVES = {
    # The official bucket's custom hostname currently has a mismatched TLS
    # certificate. Path-style S3 HTTPS verifies correctly for the same bucket.
    "coco_annotations": "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "questions_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "questions_val": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "annotations_val": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "pairs_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "pairs_val": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}
FILES = {
    "captions_train": "captions_train2014.json", "captions_val": "captions_val2014.json",
    "instances_train": "instances_train2014.json", "instances_val": "instances_val2014.json",
    "questions_train": "v2_OpenEnded_mscoco_train2014_questions.json",
    "questions_val": "v2_OpenEnded_mscoco_val2014_questions.json",
    "annotations_train": "v2_mscoco_train2014_annotations.json",
    "annotations_val": "v2_mscoco_val2014_annotations.json",
    "pairs_train": "v2_mscoco_train2014_complementary_pairs.json",
    "pairs_val": "v2_mscoco_val2014_complementary_pairs.json",
}


def ranged_download(url, path, length, etag, workers=8, chunk_size=4 * 1024 * 1024):
    """Resume verified byte ranges without deleting earlier incomplete transfers."""
    path = Path(path)
    pieces = path.with_suffix(path.suffix + ".chunks")
    pieces.mkdir(parents=True, exist_ok=True)
    provenance = pieces / "identity.json"
    identity = {"url": url, "length": length, "etag": etag, "chunk_size": chunk_size}
    if provenance.exists() and read_json(provenance) != identity:
        raise ValueError("Download chunk source identity changed")
    write_json(provenance, identity)
    spans = [(start, min(start + chunk_size, length) - 1) for start in range(0, length, chunk_size)]

    def transfer(span):
        start, end = span
        target = pieces / f"{start:012d}-{end:012d}.bin"
        if target.exists():
            if target.stat().st_size != end - start + 1:
                raise ValueError(f"Invalid completed byte range: {target}")
            return target
        temporary = target.with_suffix(f".partial-{time.time_ns()}")
        headers = {"User-Agent": "ATML-PA3", "Range": f"bytes={start}-{end}"}
        if etag:
            headers["If-Match"] = etag
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{length}":
                raise ValueError("Server did not honor the requested byte range")
            shutil.copyfileobj(response, stream, length=64 * 1024)
        if temporary.stat().st_size != end - start + 1:
            raise ValueError("Incomplete range transfer preserved for inspection")
        temporary.rename(target)
        return target

    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(pool.map(transfer, spans))
    assembled = path.with_suffix(path.suffix + f".assembled-{time.time_ns()}")
    with assembled.open("xb") as stream:
        for piece in completed:
            with piece.open("rb") as source:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
    if assembled.stat().st_size != length:
        raise ValueError("Assembled download has incorrect size")
    if path.exists():
        raise FileExistsError(path)
    assembled.rename(path)
    return path


def download(url, path, force_ranges=False):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        request = urllib.request.Request(url, headers={"User-Agent": "ATML-PA3"}, method="HEAD")
        with urllib.request.urlopen(request, timeout=45) as response:
            length = int(response.headers.get("Content-Length", 0))
            etag = response.headers.get("ETag")
            ranges = response.headers.get("Accept-Ranges") == "bytes"
        if (length > 16 * 1024 * 1024 or force_ranges) and ranges:
            return ranged_download(url, path, length, etag, chunk_size=1024 * 1024 if force_ranges else 4 * 1024 * 1024)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        partial = path.with_suffix(path.suffix + f".part-retry-{time.time_ns()}")
    request = urllib.request.Request(url, headers={"User-Agent": "ATML-PA3"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as stream:
        expected = int(response.headers.get("Content-Length", 0))
        shutil.copyfileobj(response, stream, length=64 * 1024)
    if expected and partial.stat().st_size != expected:
        raise ValueError("Truncated public download preserved without publishing final file")
    partial.rename(path)
    return path


def extract_metadata(output_dir, require_all=True):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, url in ARCHIVES.items():
        archive = root / "archives" / f"{name}.zip"
        if not archive.exists():
            if require_all:
                raise FileNotFoundError(archive)
            continue
        provenance[name] = {"url": url, "sha256": sha256(archive)}
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                filename = Path(member.filename).name
                if filename not in FILES.values():
                    continue
                destination = root / filename
                if destination.exists():
                    continue
                # Extract by validated basename, never trust archive directory paths.
                with source.open(member) as reader, destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
    paths = {key: str(root / filename) for key, filename in FILES.items()}
    if require_all and not all(Path(path).exists() for path in paths.values()):
        raise ValueError("Official archives did not contain all expected metadata files")
    write_json(root / "sources.json", provenance)
    write_json(root / "paths.json", paths)
    return paths


def metadata(output_dir):
    root = Path(output_dir)
    for name, url in ARCHIVES.items():
        download(url, root / "archives" / f"{name}.zip")
    return extract_metadata(root)


def images(manifest_path, output_dir, workers=8):
    from PIL import Image
    root = Path(output_dir)
    unique = {}
    for items in read_json(manifest_path).values():
        for item in items:
            for row in item.get("examples", [item]):
                unique[(row["split"], row["image_file"])] = row
    def transfer(key):
        split, filename = key
        if Path(filename).name != filename or split not in {"train2014", "val2014"}:
            raise ValueError("Unsafe image path in manifest")
        url = f"https://s3.us-east-1.amazonaws.com/images.cocodataset.org/{split}/{filename}"
        for attempt in range(3):
            try:
                path = download(url, root / split / filename)
                break
            except urllib.error.HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)
        with Image.open(path) as image:
            image.verify()
        return path
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for number, _ in enumerate(pool.map(transfer, sorted(unique)), start=1):
            if number % 100 == 0:
                print(f"[Task2] Verified selected images: {number}/{len(unique)}", flush=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return {"images": len(unique), "directory": str(root)}
