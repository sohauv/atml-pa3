"""Restore a hash-pinned checkpoint from extracted or zipped Kaggle inputs."""


def restore_verified_inputs(input_root, destination, checkpoint_name, expected):
    import hashlib
    from pathlib import Path, PurePosixPath
    import shutil
    import zipfile

    input_root, destination = Path(input_root), Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing recovery directory")
    relative = {}
    for name, digest in expected.items():
        parts = PurePosixPath(name).parts
        if len(parts) < 2 or ".." in parts or PurePosixPath(name).is_absolute():
            raise ValueError("Unsafe checkpoint manifest path")
        relative[str(PurePosixPath(*parts[1:]))] = digest
    marker = checkpoint_name + "/complete.json"
    if marker not in relative:
        raise ValueError("Checkpoint manifest has no complete marker")
    candidates = [("directory", p.parent.parent, "")
                  for p in sorted(input_root.rglob("complete.json"))
                  if p.parent.name == checkpoint_name]
    for archive in sorted(input_root.rglob("*.zip")):
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if name == marker or name.endswith("/" + marker):
                    candidates.append(("zip", archive, name[:-len(marker)]))
    if not candidates:
        raise ValueError(f"No extracted or zipped complete {checkpoint_name} input found")
    # Every duplicate must match the entire pinned manifest before any write.
    for kind, location, prefix in candidates:
        if kind == "directory":
            for name, digest in relative.items():
                file = location / name
                if not file.is_file():
                    raise ValueError(f"Checkpoint file missing: {file}")
                with file.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != digest:
                    raise ValueError(f"Checkpoint checksum mismatch: {file}")
        else:
            with zipfile.ZipFile(location) as source:
                for name, digest in relative.items():
                    with source.open(prefix + name) as stream:
                        actual = hashlib.file_digest(stream, "sha256").hexdigest()
                    if actual != digest:
                        raise ValueError(f"Checkpoint checksum mismatch: {location}:{prefix + name}")
    kind, location, prefix = candidates[0]
    destination.mkdir(parents=True)
    # Copy only pinned files; complete markers come after their recovery payloads.
    order = sorted(relative, key=lambda name: (name.endswith("/complete.json"), name))
    if kind == "directory":
        for name in order:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with (location / name).open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    else:
        with zipfile.ZipFile(location) as source:
            for name in order:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(prefix + name) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    print("VERIFIED_INPUT_COPIES", len(candidates), flush=True)
    return destination / checkpoint_name
