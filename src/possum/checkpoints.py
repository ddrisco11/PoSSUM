"""Atomic, durable local checkpoint replacement."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
            # macOS full flush requests physical storage synchronization.
            if os.sys.platform == 'darwin':
                import fcntl
                fcntl.fcntl(stream.fileno(), 51)  # F_FULLFSYNC
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_scores(path, values, ids, elapsed):
    atomic_write(path, lambda f: np.savez_compressed(f, values=np.asarray(values), ids=np.asarray(ids), chunk_seconds=np.asarray(elapsed)))


def save_json(path, value):
    atomic_write(path, lambda f: f.write(json.dumps(value, indent=2).encode()))
