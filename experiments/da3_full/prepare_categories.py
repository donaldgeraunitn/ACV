"""Prepare 17 WildRGB-D training sequences per category and six cellphone validation sequences."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

from prepare_real import PAIRS_PER_SCENE, build_scene, sample_evenly, write_json
from wild_archive import REVISION, open_archive

CATEGORIES = ("cellphone", "bucket", "apple", "car")


def select_pairs(archive, category, split, names, count, output):
    """Use evenly spaced official scene IDs with four calibrated pairs per sequence."""
    preferred = sample_evenly(sorted(names), count)
    ordered = preferred + [name for name in sorted(names) if name not in preferred]
    pairs = []
    for name in ordered:
        if Path(name).parent.as_posix() != "scenes":
            raise ValueError(f"Invalid scene name in the official split: {name}")
        bare_id = Path(name).name
        scene_id = f"{category}__{bare_id}" if split == "train" else bare_id
        frame_dir = bare_id if category == "cellphone" else scene_id
        rows = build_scene(
            archive, f"{category}/{name}", split, output, allow_incomplete=True, scene_id=frame_dir
        )
        if len(rows) == PAIRS_PER_SCENE:
            for row in rows:
                row["scene"] = scene_id
                if split == "train":
                    row["category"] = category
            pairs.extend(rows)
            print(f"{category} {split}: {len(pairs) // PAIRS_PER_SCENE}/{count}", flush=True)
        if len(pairs) == count * PAIRS_PER_SCENE:
            return pairs
    raise ValueError(f"Too few eligible {split} scenes in {category}")


def prepare_category(category, archives, output):
    """Read camera geometry and selected images from the official category archive."""
    path = archives / f"{category}.zip"
    handle = ZipFile(path) if path.exists() else open_archive(category, archives / "members")
    with handle as archive:
        raw = archive.read(f"{category}/nvs_list.json")
        split = json.loads(raw)
        if set(split["train"]) & set(split["val"]):
            raise ValueError(f"Overlapping train and validation scenes in {category}")
        train = select_pairs(archive, category, "train", split["train"], 17, output)
        val = (
            select_pairs(archive, category, "val", split["val"], 6, output)
            if category == "cellphone"
            else []
        )
    return train, val, hashlib.sha256(raw).hexdigest()


def prepare(archives, output):
    """Write a 68-scene training split and the original 24 real validation pairs."""
    output.mkdir(parents=True, exist_ok=True)
    train, validation, split_hashes = [], [], {}
    with ThreadPoolExecutor(max_workers=len(CATEGORIES)) as pool:
        jobs = {c: pool.submit(prepare_category, c, archives, output) for c in CATEGORIES}
        for category, job in jobs.items():
            pairs, val, split_hashes[category] = job.result()
            train.extend(pairs)
            validation.extend(val)
    if len({p["source"] for p in train + validation}) != 296:
        raise ValueError("Expected 272 distinct training and 24 distinct validation images")
    write_json(
        output / "manifest.json",
        {
            "dataset": "WildRGB-D",
            "archive_revision": REVISION,
            "official_split_sha256": split_hashes,
            "depth_units": "mm",
            "pairs": train + validation,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives", type=Path, required=True, help="ZIP archives and range-download cache"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archives, args.output)
