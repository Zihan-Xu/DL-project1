"""Audit the competition files, cache decoded images and freeze the labeled split."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "competition"
CACHE = ROOT / ".cache" / "baseline"
OUT = ROOT / "baseline"
SEED = 42


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def read_table(read_file, name, fields):
    rows = list(csv.DictReader(read_file(name).decode("utf-8-sig").splitlines()))
    if not rows or list(rows[0]) != fields:
        raise ValueError(f"Unexpected columns in {name}")
    ids = [int(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in {name}")
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--zip", type=Path, help="Also verify the exact bytes against this source archive")
    args = cli.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(args.zip)) if args.zip else None
        checked = set()

        def read_file(name):
            content = (DATA / name).read_bytes()
            if archive and content != archive.read(name):
                raise ValueError(f"Archive/workspace mismatch: {name}")
            checked.add(name)
            return content

        train = read_table(read_file, "train_labels.csv", ["id", "filename", "label"])
        test = read_table(read_file, "test.csv", ["id", "filename"])
        sample = read_table(read_file, "sample_submission.csv", ["id", "label"])
        if [x["id"] for x in sample] != [x["id"] for x in test]:
            raise ValueError("Test IDs and sample-submission IDs differ")
        labels = np.array([int(row["label"]) for row in train], dtype=np.int64)
        if set(labels) != set(range(10)):
            raise ValueError("Expected integer classes 0 through 9")

        fingerprints = {}
        for name, rows in [("train", train), ("test", test)]:
            folder = f"{name}_images"
            names = [row["filename"] for row in rows]
            if len(set(names)) != len(rows) or set(names) != {p.name for p in (DATA / folder).iterdir()}:
                raise ValueError(f"Image inventory differs from {name} CSV")
            pixels = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
            hashes = []
            for i, filename in enumerate(names):
                with Image.open(io.BytesIO(read_file(f"{folder}/{filename}"))) as im:
                    if (im.format, im.size, im.mode) != ("PNG", (32, 32), "RGB"):
                        raise ValueError(f"Unexpected image format: {filename}")
                    pixels[i] = np.asarray(im)
                hashes.append(hashlib.sha256(pixels[i].tobytes()).hexdigest())
            np.save(CACHE / f"{name}.npy", pixels)
            fingerprints[name] = hashes

        if archive:
            members = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(members) != len(set(members)) or set(members) != checked:
                raise ValueError("Unexpected, duplicate or missing files in archive")

    # Keep each exact pixel array on at most one side of any evaluation boundary.
    groups = defaultdict(list)
    for i, digest in enumerate(fingerprints["train"]):
        groups[digest].append(i)
    official = set(fingerprints["test"])
    split = np.full(len(train), "", dtype=object)
    for digest, group in groups.items():
        if digest in official:
            split[group] = "excluded_official_overlap"
        elif len(set(labels[group])) > 1:
            split[group] = "excluded_conflicting_labels"
        else:
            split[group[0]] = "eligible"
            split[group[1:]] = "excluded_duplicate"
    rng = np.random.default_rng(SEED)
    for label in range(10):
        idx = rng.permutation(np.flatnonzero((labels == label) & (split == "eligible")))
        n = round(len(idx) * 0.1)
        split[idx[:n]] = "local_test"
        split[idx[n:2*n]] = "validation"
        split[idx[2*n:]] = "train"
    with (OUT / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "filename", "label", "split", "pixel_sha256"])
        writer.writerows((row["id"], row["filename"], row["label"], split[i], fingerprints["train"][i])
                         for i, row in enumerate(train))
    # Pixel population statistics, accumulated in float64 on training rows only.
    train_idx = np.flatnonzero(split == "train")
    raw = np.load(CACHE / "train.npy", mmap_mode="r")
    sums, squares = np.zeros(3), np.zeros(3)
    for chunk in np.array_split(train_idx, max(1, len(train_idx) // 256)):
        x = raw[chunk].astype(np.float64) / 255.0
        sums += x.sum(axis=(0, 1, 2))
        squares += np.square(x).sum(axis=(0, 1, 2))
    count = len(train_idx) * 32 * 32
    mean = sums / count
    audit = {
        "archive_sha256": sha256(args.zip) if args.zip else None,
        "archive_files_byte_compared": len(checked) if args.zip else None,
        "labeled_images": len(train), "competition_images": len(test),
        "format": "32 x 32 RGB PNG", "official_test_has_labels": False,
        "sample_and_test_id_order_identical": True,
        "labeled_duplicate_pixel_groups": sum(len(g) > 1 for g in groups.values()),
        "conflicting_label_groups": sum(len(set(labels[g])) > 1 for g in groups.values()),
        "competition_duplicate_extra_rows": len(test) - len(official),
        "labeled_competition_shared_pixel_groups": len(set(groups) & official),
        "seed": SEED, "split_counts": dict(Counter(split)),
        "class_counts": {s: np.bincount(labels[split == s], minlength=10).tolist()
                         for s in ["train", "validation", "local_test"]},
        "label_counts_before_exclusions": np.bincount(labels, minlength=10).tolist(),
        "train_rgb_mean": mean.tolist(), "train_rgb_std": np.sqrt(squares / count - mean**2).tolist(),
        "split_sha256": sha256(OUT / "splits.csv"),
        "csv_sha256": {name: sha256(DATA / name) for name in ["train_labels.csv", "test.csv", "sample_submission.csv"]},
    }
    write_json(OUT / "data_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
