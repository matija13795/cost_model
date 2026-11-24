"""Retain a verified, non-overwriting project copy of a supplied raw dataset."""
import argparse
import os
from pathlib import Path
import shutil
import time
import fcntl
from ..io_utils import sha256, atomic_json


def retain(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = source.stat()
        digest = sha256(source)
        if destination.exists():
            if sha256(destination) != digest:
                raise ValueError('Project copy differs from supplied source; refusing to overwrite')
        else:
            temporary = destination.with_name(destination.name + f'.copying-{os.getpid()}')
            with open(source, 'rb') as src, open(temporary, 'xb') as dst:
                shutil.copyfileobj(src, dst, length=8*1024**2)
                dst.flush(); os.fsync(dst.fileno())
            if sha256(temporary) != digest:
                raise ValueError('Copied raw file failed checksum verification')
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed while copying')
            os.link(temporary, destination)
            # The extra link preserves the transfer artifact without duplicating data blocks.
        record = dict(source=str(source), file=str(destination), sha256=digest,
                      bytes=before.st_size, verified_unix=time.time())
        audit = destination.with_suffix('.json')
        if not audit.exists():
            atomic_json(record, audit)
        return record


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    print(retain(a.source,a.destination),flush=True)
