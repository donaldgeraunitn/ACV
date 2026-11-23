"""Create reversed training pairs with the frozen stage-one teacher."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import FullDA3Predictor
from PIL import Image
from render import render_views


def digest(path):
    """Hash the stage-one teacher checkpoint."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_png(array, path):
    """Publish only complete teacher renderings."""
    pending = path.with_suffix(".partial")
    Image.fromarray(array).save(pending, format="PNG")
    pending.replace(path)


def main():
    """Synthesize new source views and reverse the camera transformation."""
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("manifest", "cache", "checkpoint", "sharp-checkpoint", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--resolution", type=int, default=384)
    args = p.parse_args()
    root = args.manifest.parent.resolve()
    if args.output.parent.resolve() != root:
        raise ValueError("Put the pseudo-pair manifest alongside the real-frame manifest")
    chk = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if chk["stage"] != 1 or chk["variant"] != "learned":
        raise ValueError("Expected a stage-one teacher")
    model = FullDA3Predictor(args.sharp_checkpoint)
    model.load_trainable_state_dict(chk["trainable"])
    model.requires_grad_(False).eval().to("cuda")
    pairs = json.loads(args.manifest.read_text())["pairs"]
    teacher_hash = digest(args.checkpoint)
    camera_hash = digest(args.manifest)
    stem = Path(f"pseudo-{teacher_hash[:16]}-{camera_hash[:16]}-r{args.resolution}")
    records = []
    torch.set_num_threads(4)
    for i, row in enumerate(pairs):
        if row["split"] != "train":
            continue
        src = Path(row["source"])
        pseudo = stem / row["scene"] / (str(i) + ".png")
        depth = pseudo.with_name(pseudo.stem + "-depth.png")
        depth_real = pseudo.with_name(pseudo.stem + "-real-depth.png")
        Ks, Kt, t = (
            torch.as_tensor(row[k], dtype=torch.float32, device="cuda")
            for k in ("K_source", "K_target", "target_from_source")
        )
        if not all((root / name).exists() for name in (pseudo, depth, depth_real)):
            with Image.open(root / src) as im:
                rgb = np.array(im.convert("RGB"))
            h, w = rgb.shape[:2]
            image = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1)[None].float() / 255
            image = F.interpolate(
                image, size=(args.resolution, args.resolution), mode="bilinear", align_corners=True
            )
            cached = torch.load(
                args.cache / src.with_suffix(".pt"), map_location="cpu", weights_only=True
            )
            gs, _ = model(image, cached["depth"], cached["features"])
            v = render_views(
                gs,
                Ks,
                (h, w),
                torch.stack((Kt, Ks)),
                torch.stack((t, torch.eye(4, device="cuda"))),
                (h, w),
            )
            predicted = (v.rgb[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
            (root / pseudo).parent.mkdir(exist_ok=True, parents=True)
            for vi, path in enumerate((depth, depth_real)):
                z = v.depth[vi, 0]
                valid = (v.alpha[vi, 0] > 0.5) & torch.isfinite(z) & (z > 0) & (z < 65.535)
                mm = torch.where(valid, z * 1000, 0).round().cpu().numpy().astype(np.uint16)
                save_png(mm, root / path)
            save_png(predicted, root / pseudo)
        records.append(
            dict(
                scene=row["scene"],
                split="train",
                source=pseudo.as_posix(),
                target=src.as_posix(),
                source_depth=depth.as_posix(),
                target_depth=depth_real.as_posix(),
                K_source=row["K_target"],
                K_target=row["K_source"],
                target_from_source=np.linalg.inv(row["target_from_source"]).tolist(),
            )
        )
        print(f"Prepared teacher pair {len(records)}", flush=True)
    data = dict(
        pairs=records,
        depth_units="mm",
        stage=2,
        teacher_sha256=teacher_hash,
        real_cameras_sha256=camera_hash,
        variant=chk["variant"],
        protocol="Source is teacher pseudo-novel view. Target is its original real photograph. "
        "Both depths come from the frozen teacher. No paired real target or sensor depth used.",
    )
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(args.output)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
