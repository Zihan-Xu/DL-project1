"""Fine-tune the selected CIFAR ResNet on manually reviewed foreground/background pairs."""
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import ResNet20
from prepare_data import OUT, ROOT, sha256, write_json
from run_baseline import arrays, evaluate, normalize, normalizer, predict, rows, write_rows
from run_widths import STUDY as WIDTH_STUDY, directory as width_directory


STUDY = ROOT / "experiments/paired_posttraining"
QA = ROOT / "output/augmentations/background_qa"
POOL = ROOT / ".cache/augmentations"
ARMS = ("A_sft", "B_supcon")
EPOCHS, SOURCE_BATCH, SEED = 10, 128, 42
LR, MOMENTUM, WEIGHT_DECAY = 0.001, 0.9, 1e-4
TEMPERATURE, CONTRAST_WEIGHT = 0.1, 0.1


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parent_checkpoint(split_hash):
    selection = read_json(WIDTH_STUDY / "selection.json")
    selected = selection["selected"]
    path = width_directory(selected["width"]) / "resnet20.pt"
    if selection["split_sha256"] != split_hash or sha256(path) != selected["checkpoint_sha256"]:
        raise ValueError("Selected width, checkpoint or split has changed")
    return path, selected


def reviewed_indices(train_rows, split_hash):
    pool = read_json(QA / "pool.json")
    if pool["split_sha256"] != split_hash:
        raise ValueError("Background pool uses another split")
    for name, digest in pool["cache_sha256"].items():
        if sha256(POOL / name) != digest:
            raise ValueError(f"Background cache changed: {name}")
    accepted = {r["source_id"]: r for r in rows(QA / "accepted.csv")}
    positions = {r["id"]: i for i, r in enumerate(train_rows)}
    valid = np.load(POOL / "valid.npy")
    review = rows(STUDY / "pair_review.csv")
    if len({r["source_id"] for r in review}) != len(review):
        raise ValueError("Duplicate source in manual review")
    keep = []
    for record in review:
        key = record["source_id"]
        source = positions[key]
        if (not valid[source] or record["donor_id"] != accepted[key]["donor_id"]
                or record["donor_id"] not in positions or record["label"] != train_rows[source]["label"]
                or record["keep"] not in ("0", "1")):
            raise ValueError(f"Invalid reviewed pair: {key}")
        if record["keep"] == "1":
            keep.append(source)
    return np.asarray(keep, dtype=np.int64)


def batches(other, reviewed, rng):
    steps = math.ceil((len(other) + len(reviewed)) / SOURCE_BATCH)
    originals = np.array_split(rng.permutation(other), steps)
    pairs = np.array_split(rng.permutation(reviewed), steps)
    if not all(map(len, pairs)):
        raise ValueError("Need at least one reviewed pair per batch")
    return zip(originals, pairs)


def paired_supcon(features, labels, pair_count):
    # Batch tails are [reviewed originals, their composites] in the same order.
    vectors = F.normalize(features.float(), dim=1)
    indices = torch.arange(len(labels), device=labels.device)
    anchors = indices[-2 * pair_count:]
    mates = anchors.roll(pair_count)
    self_mask = anchors[:, None] == indices[None, :]
    scores = (vectors[anchors] @ vectors.T) / TEMPERATURE
    log_z = scores.masked_fill(self_mask, -torch.inf).logsumexp(dim=1)
    positive = (labels[anchors, None] == labels[None, :]) & ~self_mask
    class_score = (scores * positive).sum(dim=1) / positive.sum(dim=1)
    mate_score = scores.gather(1, mates[:, None]).squeeze(1)
    # Half of the target mass on the known mate, half shared by all same-label positives.
    return (log_z - (mate_score + class_score) * 0.5).mean()


def train(arm, device, manifest, audit):
    split_hash = audit["split_sha256"]
    path, selected = parent_checkpoint(split_hash)
    parent = torch.load(path, map_location="cpu", weights_only=True)
    images, labels, train_rows = arrays("train", manifest, device)
    val, val_labels, _ = arrays("validation", manifest, device)
    reviewed = reviewed_indices(train_rows, split_hash)
    other = np.setdiff1d(np.arange(len(labels)), reviewed)
    swaps = torch.from_numpy(np.load(POOL / "swaps.npy")).to(device)
    norm = normalizer(parent["mean"], parent["std"], device)
    model = ResNet20(selected["width"]).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(parent["model_state"])
    backbone = nn.Sequential(*list(model.children())[:-1])
    head = model[-1]
    score = int((predict(model, val, norm, SOURCE_BATCH).argmax(1) == val_labels).sum())
    if score != selected["validation_correct"]:
        raise ValueError("Parent validation score did not reproduce")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    weight = CONTRAST_WEIGHT if arm == "B_supcon" else 0.0
    cfg = dict(architecture=parent["config"]["architecture"], width_multiplier=selected["width"],
               arm=arm, seed=SEED, epochs=EPOCHS, batch_size=SOURCE_BATCH, learning_rate=LR,
               momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, scheduler="cosine", temperature=TEMPERATURE,
               contrast_weight=weight, mate_target_mass=0.5, additional_augmentation="none")
    meta = dict(config=cfg, parent_checkpoint=str(path.relative_to(ROOT)), parent_epoch=parent["epoch"],
                parent_sha256=selected["checkpoint_sha256"], parent_validation_correct=score,
                split_sha256=split_hash, review_sha256=sha256(STUDY / "pair_review.csv"),
                background_sha256=read_json(QA / "pool.json")["cache_sha256"],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                original_images_per_epoch=len(other) + len(reviewed), composites_per_epoch=len(reviewed),
                steps_per_epoch=math.ceil(len(labels) / SOURCE_BATCH), mean=parent["mean"], std=parent["std"],
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                torch_version=str(torch.__version__), training_bfloat16=use_bf16)
    folder = STUDY / arm
    folder.mkdir(exist_ok=False)
    write_json(folder / "config.json", meta)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    rng = np.random.default_rng(SEED)
    best = -1
    start = time.monotonic()
    with (folder / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "lr", "train_ce", "train_supcon", "validation_correct",
                         "validation_accuracy", "validation_loss", "batch_plan_sha256", "seconds"])
        for epoch in range(1, EPOCHS + 1):
            t0 = time.monotonic()
            model.train()
            plan_hash = hashlib.sha256()
            total_ce = torch.zeros((), device=device)
            total_con = torch.zeros((), device=device)
            seen = 0
            for plain_ids, paired_ids in batches(other, reviewed, rng):
                source_ids = np.concatenate((plain_ids, paired_ids))
                plan_hash.update(np.asarray([len(plain_ids), len(paired_ids)], dtype="<i8").tobytes())
                plan_hash.update(source_ids.astype("<i8", copy=False).tobytes())
                ix = torch.as_tensor(source_ids, device=device)
                k = len(paired_ids)
                batch_y = torch.cat((labels[ix], labels[ix[-k:]]))
                batch_x = normalize(torch.cat((images[ix], swaps[ix[-k:]])), *norm)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    features = backbone(batch_x)
                    logits = head(features)
                    ce = F.cross_entropy(logits, batch_y)
                loss = ce
                if weight:
                    con = paired_supcon(features, batch_y, k)
                    loss = loss + weight * con
                    total_con += con.detach() * k
                loss.backward()
                optimizer.step()
                total_ce += ce.detach() * len(batch_y)
                seen += len(batch_y)
            logits = predict(model, val, norm, SOURCE_BATCH)
            correct = int((logits.argmax(1) == val_labels).sum())
            accuracy = correct / len(val_labels)
            val_loss = F.cross_entropy(logits, val_labels).item()
            ce_mean = total_ce.item() / seen
            con_mean = total_con.item() / len(reviewed)
            if seen != len(labels) + len(reviewed) or not np.isfinite([val_loss, ce_mean, con_mean]).all():
                raise ValueError("Invalid epoch counts or losses")
            writer.writerow([epoch, optimizer.param_groups[0]["lr"], ce_mean, con_mean if weight else "",
                             correct, accuracy, val_loss, plan_hash.hexdigest(), time.monotonic() - t0])
            stream.flush()
            if correct > best:
                best = correct
                tmp = folder / "checkpoint.tmp"
                torch.save(dict(**meta, epoch=epoch, validation_accuracy=accuracy, model_state=model.state_dict()), tmp)
                tmp.replace(folder / "resnet20.pt")
            schedule.step()
            print(f"{arm} {epoch:02}/{EPOCHS}: {correct}/{len(val_labels)} = {accuracy:.2%}", flush=True)
    write_json(folder / "run.json", dict(**meta, training_seconds=time.monotonic() - start))


def compare(device, manifest, audit):
    parent_path, selected = parent_checkpoint(audit["split_sha256"])
    images, labels, val_rows = arrays("validation", manifest, device)
    records, predictions = [], []
    plans = []
    for name, path in [("parent", parent_path)] + [(arm, STUDY / arm / "resnet20.pt") for arm in ARMS]:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = ResNet20(selected["width"]).to(device=device, memory_format=torch.channels_last)
        model.load_state_dict(ckpt["model_state"])
        pred = predict(model, images, normalizer(ckpt["mean"], ckpt["std"], device), SOURCE_BATCH).argmax(1)
        correct = int((pred == labels).sum())
        if correct / len(labels) != ckpt["validation_accuracy"]:
            raise ValueError(f"Validation score mismatch for {name}")
        if name in ARMS:
            run = read_json(path.parent / "run.json")
            history = rows(path.parent / "history.csv")
            if (run["parent_sha256"] != selected["checkpoint_sha256"] or
                    [int(r["epoch"]) for r in history] != list(range(1, EPOCHS + 1)) or
                    ckpt["epoch"] != int(max(history, key=lambda r: int(r["validation_correct"]))["epoch"])):
                raise ValueError(f"Inconsistent training records for {name}")
            plans.append([r["batch_plan_sha256"] for r in history])
        records.append(dict(arm=name, epoch=ckpt["epoch"], validation_correct=correct,
                            validation_accuracy=correct / len(labels), checkpoint_sha256=sha256(path)))
        predictions.append(pred.tolist())
    if plans[0] != plans[1]:
        raise ValueError("A and B used different batches")
    write_rows(STUDY / "comparison.csv", list(records[0]), [r.values() for r in records])
    write_rows(STUDY / "validation_predictions.csv", ["id", "label", "parent", *ARMS],
               zip([r["id"] for r in val_rows], labels.tolist(), *predictions))
    selected_arm = max(records[1:], key=lambda r: r["validation_correct"])["arm"]
    best_overall = max(records, key=lambda r: r["validation_correct"])["arm"]
    write_json(STUDY / "selection.json", dict(selected_posttraining_arm=selected_arm, best_overall=best_overall,
               rule="max validation correct; tie order parent, A_sft, B_supcon", paired_batches_verified=True))
    print(f"Selected post-training arm: {selected_arm}; best including parent: {best_overall}", flush=True)
    evaluate(device, audit, manifest, out=STUDY / selected_arm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[*ARMS, "compare"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    audit = read_json(OUT / "data_audit.json")
    if sha256(OUT / "splits.csv") != audit["split_sha256"]:
        raise ValueError("Split manifest has changed")
    manifest = rows(OUT / "splits.csv")
    device = torch.device(args.device)
    if args.action == "compare":
        compare(device, manifest, audit)
    else:
        train(args.action, device, manifest, audit)
