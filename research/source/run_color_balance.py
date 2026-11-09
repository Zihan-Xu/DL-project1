"""Paired label-only versus within-label red/blue-tone balanced sampling."""
import argparse
import csv
import hashlib
import json
from contextlib import ExitStack

import numpy as np
import torch
from torch.nn import functional as F

from model import ResNet20
from prepare_data import CACHE, OUT, ROOT, sha256, write_json
from run_baseline import arrays, evaluate, normalize, normalizer, predict, rows, train, write_rows


STUDY = ROOT / "experiments" / "balanced_sampling" / "color_tone"
SEEDS = (42, 43, 44)
ARMS = ("B_label", "B_tone")
PROTOCOL = "label_tone_v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_inputs():
    audit = read_json(OUT / "data_audit.json")
    if sha256(OUT / "splits.csv") != audit["split_sha256"]:
        raise ValueError("Frozen split manifest has changed")
    manifest = rows(OUT / "splits.csv")
    source = np.load(CACHE / "train.npy", mmap_mode="r")
    indices = np.array([i for i, row in enumerate(manifest) if row["split"] == "train"])
    labels = np.array([int(manifest[i]["label"]) for i in indices])
    if not np.array_equal(np.bincount(labels), np.full(10, 4000)):
        raise ValueError("Expected 4,000 training originals in each label")
    return audit, manifest, source, indices, labels


def tone_features(source, indices):
    result = []
    for start in range(0, len(indices), 256):
        images = source[indices[start:start + 256]].astype(np.float32) / 255
        rgb = images.mean(axis=(1, 2), dtype=np.float64)
        result.append(rgb[:, 0] - rgb[:, 2])
    return np.concatenate(result)


def training_statistics(source, indices):
    sums, squares = np.zeros(3), np.zeros(3)
    for start in range(0, len(indices), 256):
        images = source[indices[start:start + 256]].astype(np.float64) / 255
        sums += images.sum(axis=(0, 1, 2))
        squares += np.square(images).sum(axis=(0, 1, 2))
    count = len(indices) * 32 * 32
    mean = sums / count
    return mean.tolist(), np.sqrt(squares / count - mean ** 2).tolist()


def choose_tone_subset(class_order, bins):
    # Keep the common 1,000 originals, then fill the least populated feasible bin.
    selected = class_order[:1000].tolist()
    counts = np.bincount(bins[selected], minlength=3)
    remaining = [class_order[1000:][bins[class_order[1000:]] == b] for b in range(3)]
    used = np.zeros(3, dtype=int)
    cursor = 0
    while len(selected) < 2000:
        available = [b for b in range(3) if used[b] < len(remaining[b])]
        lowest = min(counts[b] for b in available)
        chosen = next(b for b in ((cursor + offset) % 3 for offset in range(3))
                      if b in available and counts[b] == lowest)
        selected.append(int(remaining[chosen][used[chosen]]))
        used[chosen] += 1
        counts[chosen] += 1
        cursor = (chosen + 1) % 3
    selected_set = set(selected)
    return np.array([i for i in class_order if i in selected_set], dtype=np.int64)


def prepare(audit, manifest, source, train_indices, labels):
    if (STUDY / "prepared.json").exists():
        verify_prepared(audit)
        print("Existing prepared inputs verified.", flush=True)
        return
    STUDY.mkdir(parents=True, exist_ok=True)
    # Check pixels used for training and validation against the frozen audit.
    for i, row in enumerate(manifest):
        if row["split"] in ("train", "validation"):
            if digest_array(source[i]) != row["pixel_sha256"]:
                raise ValueError(f"Image cache differs from frozen original: {row['id']}")
    tone = tone_features(source, train_indices)
    edges = np.quantile(tone, [1 / 3, 2 / 3])
    bins = np.searchsorted(edges, tone, side="right")
    source_counts = np.bincount(labels * 3 + bins, minlength=30).reshape(10, 3)
    write_rows(STUDY / "training_colors.csv", ["train_index", "id", "label", "red_minus_blue", "tone_bin"],
               ((i, manifest[source_i]["id"], int(labels[i]), float(tone[i]), int(bins[i]))
                for i, source_i in enumerate(train_indices)))
    selected_counts = []
    for seed in SEEDS:
        folder = STUDY / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        permutation = np.random.default_rng(seed).permutation(len(labels))
        class_orders = [permutation[labels[permutation] == label] for label in range(10)]
        common = np.concatenate([order[:1000] for order in class_orders])
        common_set = set(common.tolist())
        np.save(folder / "common_indices.npy", common)
        mean, std = training_statistics(source, train_indices[common])
        write_json(folder / "normalization.json", {"mean": mean, "std": std,
                                                  "common_indices_sha256": sha256(folder / "common_indices.npy")})
        choices = {
            "B_label": np.stack([order[:2000] for order in class_orders]),
            "B_tone": np.stack([choose_tone_subset(order, bins) for order in class_orders]),
        }
        for arm, selection in choices.items():
            target = folder / arm
            target.mkdir(exist_ok=True)
            np.save(target / "indices.npy", selection)
            if len(np.unique(selection)) != 20000 or not np.isin(common, selection).all():
                raise ValueError("Selection violates uniqueness or common-subset constraint")
            write_rows(target / "subset.csv", ["train_index", "id", "filename", "label", "tone_bin", "common"],
                       ((int(i), manifest[train_indices[i]]["id"], manifest[train_indices[i]]["filename"],
                         int(labels[i]), int(bins[i]), bool(i in common_set))
                        for i in selection.ravel()))
            counts = np.bincount(labels[selection.ravel()] * 3 + bins[selection.ravel()], minlength=30).reshape(10, 3)
            for label in range(10):
                selected_counts.append([seed, arm, label, *counts[label].tolist()])
            cfg = read_json(ROOT / "config.json")
            cfg.update(seed=seed, width_multiplier=3, crop_padding=0,
                       study_protocol=PROTOCOL, sampling_arm=arm,
                       unique_training_images=20000, presentations_per_epoch=40000,
                       subset_sha256=sha256(target / "indices.npy"),
                       common_indices_sha256=sha256(folder / "common_indices.npy"),
                       augmentation={"name": "paired_horizontal_flip", "probability": 0.5})
            write_json(target / "config.json", cfg)
    write_rows(STUDY / "subset_color_counts.csv", ["seed", "arm", "label", "low", "middle", "high"], selected_counts)
    files = sorted(p for p in STUDY.rglob("*") if p.is_file())
    write_json(STUDY / "prepared.json", {
        "protocol": PROTOCOL, "split_sha256": audit["split_sha256"],
        "seeds": list(SEEDS), "arms": list(ARMS), "tone_bin_edges": edges.tolist(),
        "source_label_tone_counts": source_counts.tolist(),
        "cache_pixels_verified": 45000,
        "file_sha256": {str(p.relative_to(STUDY)).replace("\\", "/"): sha256(p) for p in files},
        "selection_rule": "highest mean best-validation correct across paired seeds; tie B_label; final test seed 42",
    })
    print("Prepared 6 label-balanced subsets and frozen train-only tone bins.", flush=True)


def verify_prepared(audit):
    prepared = read_json(STUDY / "prepared.json")
    if prepared["protocol"] != PROTOCOL or prepared["split_sha256"] != audit["split_sha256"]:
        raise ValueError("Prepared protocol/split mismatch")
    for name, digest in prepared["file_sha256"].items():
        if sha256(STUDY / name) != digest:
            raise ValueError(f"Prepared file changed: {name}")
    return prepared


def epoch_plan(selection, labels, seed, epoch, batch_size=128):
    # Tokens encode class, occurrence (0/1), and selection rank, identically in both arms.
    token_labels = np.repeat(np.arange(10), 4000)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 100, epoch]))
    for attempt in range(1000):
        tokens = rng.permutation(40000)
        ordered_labels = token_labels[tokens]
        batch_labels = np.arange(40000) // batch_size
        counts = np.bincount(batch_labels * 10 + ordered_labels,
                             minlength=((40000 + batch_size - 1) // batch_size) * 10).reshape(-1, 10)
        if counts.min() > 0:
            break
    else:
        raise RuntimeError("Could not produce batches covering all labels")
    occurrence = (tokens % 4000) // 2000
    indices = selection[ordered_labels, tokens % 2000]
    flip_rng = np.random.default_rng(np.random.SeedSequence([seed, 200, epoch]))
    flip_table = flip_rng.random((2, len(labels))) < 0.5
    flips = flip_table[occurrence, indices]
    if not np.array_equal(labels[indices], ordered_labels):
        raise ValueError("Selection rank does not match its class")
    unique, frequency = np.unique(indices, return_counts=True)
    if len(unique) != 20000 or not np.all(frequency == 2):
        raise ValueError("Expected exactly two presentations of each selected original")
    return indices, occurrence, flips, counts, attempt


class PairedSampling:
    def __init__(self, selection, labels, seed, device, epoch_writer, batch_writer):
        self.selection, self.labels, self.seed = selection, labels, seed
        self.epoch_writer, self.batch_writer = epoch_writer, batch_writer
        self.applied = torch.zeros((), dtype=torch.int64, device=device)
        self.flip_mask = None

    def batches(self, epoch, device, batch_size):
        indices, occurrence, flips, counts, rejected = epoch_plan(
            self.selection, self.labels, self.seed, epoch, batch_size)
        self.epoch_writer.writerow([epoch, len(np.unique(indices)), len(indices), rejected,
                                    digest_array(indices), digest_array(self.labels[indices]),
                                    digest_array(np.column_stack((indices, occurrence, flips))),
                                    int(flips.sum()), int(counts.min()), int(counts.max())])
        for number, batch_counts in enumerate(counts, 1):
            self.batch_writer.writerow([epoch, number, *batch_counts.tolist()])
        gpu_indices = torch.as_tensor(indices, device=device)
        gpu_flips = torch.as_tensor(flips, device=device)
        for start in range(0, len(indices), batch_size):
            self.flip_mask = gpu_flips[start:start + batch_size]
            yield gpu_indices[start:start + batch_size]

    def __call__(self, images, indices, norm):
        raw = images[indices]
        self.applied += self.flip_mask.sum()
        changed = torch.where(self.flip_mask[:, None, None, None], raw.flip((2,)), raw)
        return normalize(changed, *norm)


def preflight(audit, labels):
    verify_prepared(audit)
    checks = []
    for seed in SEEDS:
        plans = {}
        for arm in ARMS:
            selection = np.load(STUDY / f"seed_{seed}" / arm / "indices.npy")
            for epoch in (1, 2, 80):
                plan = epoch_plan(selection, labels, seed, epoch)
                indices, occurrence, flips, counts, rejected = plan
                if not np.all(counts.sum(axis=0) == 4000) or counts[-1].sum() != 64:
                    raise ValueError("Epoch or tail-batch quotas are wrong")
                plans[arm, epoch] = plan
                checks.append({"seed": seed, "arm": arm, "epoch": epoch,
                               "batch_count": len(counts), "minimum_label_per_batch": int(counts.min())})
        for epoch in (1, 2, 80):
            left, right = plans["B_label", epoch], plans["B_tone", epoch]
            if not np.array_equal(left[3], right[3]):
                raise ValueError("Paired batch label counts differ")
            left_flips = {(int(i), int(o)): bool(f) for i, o, f in zip(*left[:3])}
            for i, o, f in zip(*right[:3]):
                if (int(i), int(o)) in left_flips and left_flips[int(i), int(o)] != bool(f):
                    raise ValueError("Paired augmentation differs on a common image")
    write_json(STUDY / "preflight.json", {"passed": True, "checks": checks})
    print("Sampler preflight passed for all seeds, both arms, and epochs 1/2/80.", flush=True)


def train_all(device, audit, manifest, labels):
    preflight(audit, labels)
    for seed in SEEDS:
        folder = STUDY / f"seed_{seed}"
        norm = read_json(folder / "normalization.json")
        training_audit = dict(audit, train_rgb_mean=norm["mean"], train_rgb_std=norm["std"])
        for arm in ARMS:
            target = folder / arm
            cfg = read_json(target / "config.json")
            if (target / "run.json").exists():
                run = read_json(target / "run.json")
                history = rows(target / "history.csv")
                if run.get("training_seconds") is not None and len(history) == cfg["epochs"]:
                    print(f"Already completed seed={seed} arm={arm}", flush=True)
                    continue
                raise RuntimeError(f"Incomplete run needs inspection before retry: {target}")
            selection = np.load(target / "indices.npy")
            print(f"START seed={seed} arm={arm}", flush=True)
            with ExitStack() as stack:
                epoch_file = stack.enter_context((target / "sampling_epochs.csv").open("w", newline="", encoding="utf-8"))
                batch_file = stack.enter_context((target / "batch_label_counts.csv").open("w", newline="", encoding="utf-8"))
                epoch_writer, batch_writer = csv.writer(epoch_file), csv.writer(batch_file)
                epoch_writer.writerow(["epoch", "unique_images", "presentations", "rejected_permutations",
                                       "indices_sha256", "label_sequence_sha256", "augmentation_sha256",
                                       "flipped", "min_batch_label_count", "max_batch_label_count"])
                batch_writer.writerow(["epoch", "batch", *[f"label_{c}" for c in range(10)]])
                sampling = PairedSampling(selection, labels, seed, device, epoch_writer, batch_writer)
                train(cfg, device, training_audit, manifest, out=target,
                      augmentation=sampling, epoch_batches=sampling.batches)
            print(f"DONE seed={seed} arm={arm}", flush=True)


def compare(device, audit, manifest, source):
    prepared = verify_prepared(audit)
    val_images, val_y, val_rows = arrays("validation", manifest, device)
    val_indices = np.array([i for i, row in enumerate(manifest) if row["split"] == "validation"])
    val_tone = tone_features(source, val_indices)
    val_bins = np.searchsorted(prepared["tone_bin_edges"], val_tone, side="right")
    true = val_y.cpu().numpy()
    records, strata, all_predictions = [], [], {}
    for seed in SEEDS:
        paired_sampling = []
        for arm in ARMS:
            target = STUDY / f"seed_{seed}" / arm
            cfg = read_json(target / "config.json")
            run = read_json(target / "run.json")
            history = rows(target / "history.csv")
            ckpt = torch.load(target / "resnet20.pt", map_location="cpu", weights_only=True)
            if (len(history) != 80 or ckpt["config"] != cfg or run["config"] != cfg
                    or ckpt["split_sha256"] != audit["split_sha256"]
                    or run.get("training_examples_seen") != 3_200_000):
                raise ValueError(f"Incomplete or mismatched run: {target}")
            best = max(history, key=lambda row: float(row["validation_accuracy"]))
            model = ResNet20(3).to(device=device, memory_format=torch.channels_last)
            model.load_state_dict(ckpt["model_state"])
            logits = predict(model, val_images, normalizer(ckpt["mean"], ckpt["std"], device), 128)
            predicted = logits.argmax(1).cpu().numpy()
            correct = predicted == true
            if ckpt["epoch"] != int(best["epoch"]) or float(correct.mean()) != ckpt["validation_accuracy"]:
                raise ValueError("Checkpoint validation result does not reproduce")
            confusion = np.bincount(true * 10 + predicted, minlength=100).reshape(10, 10)
            class_f1 = 2 * confusion.diagonal() / np.maximum(confusion.sum(0) + confusion.sum(1), 1)
            record = {"seed": seed, "arm": arm, "best_epoch": ckpt["epoch"],
                      "validation_correct": int(correct.sum()), "validation_accuracy": float(correct.mean()),
                      "validation_loss": F.cross_entropy(logits, val_y).item(),
                      "macro_f1": float(class_f1.mean()), "training_seconds": run["training_seconds"],
                      "checkpoint_sha256": sha256(target / "resnet20.pt")}
            records.append(record)
            write_json(target / "validation_metrics.json", dict(record, confusion_matrix=confusion.tolist()))
            all_predictions[f"{seed}_{arm}"] = predicted.tolist()
            for label in range(10):
                for color_bin in range(3):
                    mask = (true == label) & (val_bins == color_bin)
                    n = int(mask.sum())
                    strata.append([seed, arm, label, color_bin, n, int(correct[mask].sum()),
                                   float(correct[mask].mean()) if n else ""])
            sample_log = rows(target / "sampling_epochs.csv")
            batch_log = rows(target / "batch_label_counts.csv")
            if len(sample_log) != 80 or len(batch_log) != 25040:
                raise ValueError("Sampling log length is wrong")
            for item in batch_log:
                if min(int(item[f"label_{c}"]) for c in range(10)) < 1:
                    raise ValueError("A batch missed a class")
            paired_sampling.append([item["label_sequence_sha256"] for item in sample_log])
        if paired_sampling[0] != paired_sampling[1]:
            raise ValueError("Label order differs between paired runs")
    write_rows(STUDY / "comparison.csv", list(records[0]), (record.values() for record in records))
    write_rows(STUDY / "validation_color_strata.csv",
               ["seed", "arm", "label", "tone_bin", "n", "correct", "accuracy"], strata)
    write_rows(STUDY / "validation_predictions.csv", ["id", "label", "tone_bin", *all_predictions],
               zip([row["id"] for row in val_rows], true.tolist(), val_bins.tolist(), *all_predictions.values()))
    summary = {}
    for arm in ARMS:
        scores = np.array([record["validation_accuracy"] for record in records if record["arm"] == arm])
        summary[arm] = {"mean_validation_accuracy": float(scores.mean()), "std_validation_accuracy": float(scores.std(ddof=1)),
                        "seed_validation_accuracy": scores.tolist()}
    differences = np.array(summary["B_tone"]["seed_validation_accuracy"]) - np.array(summary["B_label"]["seed_validation_accuracy"])
    summary["paired_tone_minus_label_pp"] = (differences * 100).tolist()
    summary["mean_paired_delta_pp"] = float(differences.mean() * 100)
    selected = max(ARMS, key=lambda arm: sum(record["validation_correct"] for record in records if record["arm"] == arm))
    selection = {"selected_arm": selected, "selected_seed": 42,
                 "rule": prepared["selection_rule"], "summary": summary,
                 "checkpoint_sha256": sha256(STUDY / "seed_42" / selected / "resnet20.pt")}
    write_json(STUDY / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "compare", "evaluate"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    audit, manifest, source, train_indices, labels = load_inputs()
    device = torch.device(args.device)
    if args.action == "prepare":
        prepare(audit, manifest, source, train_indices, labels)
        preflight(audit, labels)
    elif args.action == "train":
        train_all(device, audit, manifest, labels)
    elif args.action == "compare":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        compare(device, audit, manifest, source)
    else:
        verify_prepared(audit)
        selection = read_json(STUDY / "selection.json")
        target = STUDY / f"seed_{selection['selected_seed']}" / selection["selected_arm"]
        if sha256(target / "resnet20.pt") != selection["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        evaluate(device, audit, manifest, out=target)


if __name__ == "__main__":
    main()
