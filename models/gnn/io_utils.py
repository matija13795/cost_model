"""Project artifact integrity and atomic writes."""
import hashlib
import json
import os
from pathlib import Path
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def annotation_hash(annotation):
    return hashlib.sha256(json.dumps(annotation, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
