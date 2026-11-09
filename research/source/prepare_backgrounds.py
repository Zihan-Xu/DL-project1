"""Build one independently checked foreground/background swap per eligible training ID."""
import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from prepare_data import CACHE, OUT, ROOT, SEED, sha256, write_json


QA = ROOT / "output" / "augmentations" / "background_qa"
POOL = ROOT / ".cache" / "augmentations"
MIN_IOU = 0.75
MAX_DONORS = 5
ITERATIONS = 4


def segment(image):
    # Same deterministic, rectangle-only initialization for raw and synthetic images.
    cv2.setRNGSeed(SEED)
    mask = np.zeros((32, 32), np.uint8)
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(image, mask, (1, 1, 30, 30), bg, fg, ITERATIONS, cv2.GC_INIT_WITH_RECT)
    binary = mask & 1
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    size = stats[largest, cv2.CC_STAT_AREA]
    if not (0.12 <= size / binary.size <= 0.75) or size < 0.8 * binary.sum():
        return None
    return (components == largest).astype(np.uint8)


def save_examples(images, masks, swaps, donors, labels, ids, scores):
    rng = np.random.default_rng(SEED)
    records = []
    for label in range(10):
        group = np.flatnonzero((labels == label) & (donors >= 0))
        examples = rng.choice(group, min(20, len(group)), replace=False)
        sheet = Image.new("RGB", (5 * 200, 4 * 76 + 22), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 2), f"Class {label} | source / mask / donor / composite", fill="black")
        for pos, i in enumerate(examples):
            x, y = (pos % 5) * 200, 22 + (pos // 5) * 76
            j = donors[i]
            parts = (images[i], masks[i] * 255, images[j], swaps[i])
            for col, part in enumerate(parts):
                thumb = Image.fromarray(part).resize((48, 48), Image.Resampling.NEAREST)
                sheet.paste(thumb, (x + 49 * col, y))
            draw.text((x, y + 49), f"{ids[i]} + {ids[j]}, IoU {scores[i]:.2f}", fill="black")
            records.append([label, ids[i], ids[j], float(scores[i])])
        sheet.save(QA / f"class_{label}.png")
    with (QA / "review_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "source_id", "donor_id", "iou"])
        writer.writerows(records)


def main():
    cv2.setNumThreads(1)
    audit = json.loads((OUT / "data_audit.json").read_text())
    if sha256(OUT / "splits.csv") != audit["split_sha256"]:
        raise ValueError("Split manifest mismatch")
    QA.mkdir(parents=True, exist_ok=True)
    POOL.mkdir(parents=True, exist_ok=True)
    if (QA / "pool.json").exists():
        raise FileExistsError("An audited background pool already exists")
    with (OUT / "splits.csv").open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    train = [i for i, r in enumerate(manifest) if r["split"] == "train"]
    images = np.load(CACHE / "train.npy", mmap_mode="r")[train]
    ids = np.array([int(manifest[i]["id"]) for i in train])
    labels = np.array([int(manifest[i]["label"]) for i in train])
    masks = np.zeros(images.shape[:3], np.uint8)
    for i, image in enumerate(images):
        mask = segment(image)
        if mask is not None:
            masks[i] = mask
        if (i + 1) % 5000 == 0:
            print(f"Segmented {i + 1}/{len(images)}", flush=True)
    eligible = np.flatnonzero(masks.any(axis=(1, 2)))
    rng = np.random.default_rng(SEED)
    swaps = np.zeros_like(images)
    donors = np.full(len(images), -1)
    scores = np.zeros(len(images), dtype=float)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    trials = 0
    for n, i in enumerate(eligible):
        source_mask = masks[i].astype(bool)
        candidates = rng.choice(eligible, min(MAX_DONORS + 1, len(eligible)), replace=False)
        for j in candidates[candidates != i][:MAX_DONORS]:
            erase = cv2.dilate(masks[j], kernel)
            # Discard pairs that would expose a large, artificially inpainted area.
            if (erase.astype(bool) & ~source_mask).mean() > 0.18:
                continue
            background = cv2.inpaint(images[j], erase, 2, cv2.INPAINT_TELEA)
            composite = np.where(source_mask[:, :, None], images[i], background)
            recovered = segment(composite)
            trials += 1
            if recovered is None:
                continue
            iou = (source_mask & recovered).sum() / (source_mask | recovered).sum()
            if iou >= MIN_IOU:
                swaps[i], donors[i], scores[i] = composite, j, iou
                break
        if (n + 1) % 5000 == 0:
            print(f"Paired {n + 1}/{len(eligible)} sources", flush=True)
    valid = donors >= 0
    if not valid.any():
        raise ValueError("No usable swaps")
    for name, data in (("swaps", swaps), ("valid", valid)):
        np.save(POOL / f"{name}.npy", data)
    with (QA / "accepted.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "source_label", "donor_id", "donor_label", "mask_iou"])
        for i in np.flatnonzero(valid):
            j = donors[i]
            writer.writerow([ids[i], labels[i], ids[j], labels[j], scores[i]])
    save_examples(images, masks, swaps, donors, labels, ids, scores)
    report = {
        "split_sha256": audit["split_sha256"], "opencv": cv2.__version__, "seed": SEED,
        "algorithm": "GrabCut, rectangle (1,1,30,30), same initialization on re-extraction",
        "iterations": ITERATIONS, "largest_component_area_fraction": [0.12, 0.75],
        "min_largest_component_share": 0.8, "max_exposed_inpaint_fraction": 0.18,
        "max_donors_per_source": MAX_DONORS, "minimum_mask_iou": MIN_IOU,
        "train_examples": len(images), "eligible_source_masks": len(eligible),
        "composites_resegmented": trials, "accepted_sources": int(valid.sum()),
        "accepted_by_class": np.bincount(labels[valid], minlength=10).tolist(),
        "training_by_class": np.bincount(labels, minlength=10).tolist(),
        "mean_accepted_mask_iou": float(scores[valid].mean()),
        "cache_sha256": {name: sha256(POOL / name) for name in ("swaps.npy", "valid.npy")},
    }
    write_json(QA / "pool.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
