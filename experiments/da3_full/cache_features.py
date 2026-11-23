"""Cache DA3 features and metric depth using the source photograph and its intrinsics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from model import FrozenDA3
from PIL import Image


def main():
    """Cache each distinct source once and invalidate it when calibration or weights change."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--da3-checkpoint", type=Path, required=True)
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()
    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = doc if isinstance(doc, list) else doc["pairs"]
    with (args.da3_checkpoint / "model.safetensors").open("rb") as stream:
        weights_id = hashlib.file_digest(stream, "sha256").hexdigest()
    sources = {}
    for pair in pairs:
        name = pair["source"]
        key = pair["K_source"]
        if name in sources and sources[name] != key:
            raise ValueError(f"Inconsistent source intrinsics for {name}")
        sources[name] = key
    net = None
    torch.set_num_threads(4)
    with torch.no_grad():
        for i, (name, k) in enumerate(sources.items(), 1):
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Expected relative image path: {name}")
            inp = args.manifest.parent / rel
            dest = args.cache / rel.with_suffix(".pt")
            with inp.open("rb") as stream:
                source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            spec = dict(
                source_hash=source_hash, k=k, process_res=args.process_res, weights=weights_id
            )
            stamp = dest.with_suffix(".json")
            if dest.exists() and stamp.exists() and json.loads(stamp.read_text()) == spec:
                continue
            if net is None:
                net = FrozenDA3(args.da3_checkpoint, "cuda", process_res=args.process_res)
            with Image.open(inp) as photo:
                rgb = np.array(photo.convert("RGB"))
            item = net.extract(rgb, float(k[0][0]), float(k[1][1]))
            if not torch.isfinite(item["depth"]).all() or not (item["depth"] > 0).all():
                raise ValueError(f"Invalid DA3 depth for {name}")
            dest.parent.mkdir(exist_ok=True, parents=True)
            tmp = dest.with_suffix(".tmp")
            torch.save(item, tmp)
            tmp.replace(dest)
            tmp = stamp.with_suffix(".tmp")
            tmp.write_text(json.dumps(spec))
            tmp.replace(stamp)
            print(f"{i}/{len(sources)} {name}", flush=True)


if __name__ == "__main__":
    main()
