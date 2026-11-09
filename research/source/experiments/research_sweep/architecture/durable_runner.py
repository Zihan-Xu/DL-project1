"""Launch the frozen runner with durable category checkpoint and CSV writers."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from experiments.research_sweep.architecture import runner

shared_atomic_json = runner.runtime.atomic_json


def temporary(path):
    return path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')


def flush_file(stream):
    stream.flush()
    os.fsync(stream.fileno())


def replace_with_retry(source, destination):
    deadline = time.monotonic() + 5.0
    delay = 0.025
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 0.25)


def save_json(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(HERE):
        return shared_atomic_json(path, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = temporary(path)
    with new.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        flush_file(stream)
    replace_with_retry(new, path)


def save_checkpoint(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to((HERE/'runs').resolve())
    new = temporary(path)
    with new.open('wb') as stream:
        torch.save(value, stream)
        flush_file(stream)
    if path.name == 'last.pt' and path.exists():
        # The prior checkpoint is immutable, because updates always replace files.
        previous = path.with_name('previous.pt')
        link = temporary(previous)
        try:
            os.link(path, link)
        except OSError:
            with path.open('rb') as src, link.open('wb') as dst:
                shutil.copyfileobj(src, dst)
                flush_file(dst)
        replace_with_retry(link, previous)
    replace_with_retry(new, path)


def save_csv(path, records, fields=None):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    new = temporary(path)
    with new.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        flush_file(stream)
    replace_with_retry(new, path)


if __name__ == '__main__':
    source = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    record = HERE/'checkpoint_io_protocol_v2.json'
    assert runner.read(record)['wrapper_sha256'] == source_hash
    runner.save = save_checkpoint
    runner.atomic_csv = save_csv
    runner.runtime.atomic_json = save_json
    sys.exit(runner.main())
