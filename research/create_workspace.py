"""Copy development sources into a fresh directory for historical experiment reproduction."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["studies", "width"],
        default="studies",
        help="studies includes the fixed project teacher; width starts a fresh width sweep",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional course-data folder to copy into the new workspace",
    )
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    assert not destination.exists(), "Use a new output directory"
    assert not destination.is_relative_to(
        ROOT.parent.resolve()
    ), "Keep runs outside the release"
    for name, record in json.loads((ROOT / "SOURCE_SHA256.json").read_text()).items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["sha256"]
        ), name
    if args.data_dir:
        data = args.data_dir.resolve()
        for name in [
            "train_images",
            "test_images",
            "train_labels.csv",
            "test.csv",
            "sample_submission.csv",
        ]:
            assert (data / name).exists(), name
        assert not destination.is_relative_to(
            data
        ), "Keep the workspace outside the input data folder"
    shutil.copytree(ROOT / "source", destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "delivery_reports").mkdir(exist_ok=True)
    if args.mode == "studies":
        shutil.copytree(ROOT / "reference", destination, dirs_exist_ok=True)
    if args.data_dir:
        target = destination / "data/competition"
        target.mkdir(parents=True)
        for name in [
            "train_images",
            "test_images",
            "train_labels.csv",
            "test.csv",
            "sample_submission.csv",
        ]:
            if (data / name).is_dir():
                shutil.copytree(data / name, target / name)
            else:
                shutil.copy2(data / name, target / name)
    print(f"Created {args.mode} workspace: {destination}")
    print(
        "Follow research/README.md from this workspace. No training has been started."
    )


if __name__ == "__main__":
    main()
