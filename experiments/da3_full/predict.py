"""Reconstruct a single photograph with DA3 and export its Gaussian scene as a PLY."""

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
from model import FrozenDA3, FullDA3Predictor
from sharp.utils.gaussians import save_ply, unproject_gaussians
from sharp.utils.io import load_rgb


def reconstruct(model, rgb, features, focal_px, resolution, device):
    """Predict NDC Gaussians and unproject them into the metrical source-camera frame."""
    height, width = rgb.shape[:2]
    tensor = torch.from_numpy(np.array(rgb, copy=True)).to(device=device, dtype=torch.float32)
    source = F.interpolate(
        tensor.permute(2, 0, 1)[None] / 255,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=True,
    )
    ndc, _ = model(source, features["depth"], features["features"])
    intrinsics = torch.tensor(
        [[focal_px, 0, width / 2, 0], [0, focal_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    intrinsics[0] *= resolution / width
    intrinsics[1] *= resolution / height
    return unproject_gaussians(
        ndc, torch.eye(4, device=device), intrinsics, (resolution, resolution)
    )


@torch.inference_mode()
def predict(args):
    """Load the learned depth head and Gaussian decoder, then reconstruct the image."""
    if args.output.suffix.lower() != ".ply":
        raise ValueError("--output must name a .ply file.")
    if args.resolution < 32 or args.resolution % 32 != 0:
        raise ValueError("--resolution must be a positive multiple of 32.")
    if not args.da3_checkpoint.is_dir():
        raise FileNotFoundError(f"DA3 checkpoint directory does not exist: {args.da3_checkpoint}")
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if state["variant"] != "learned":
        raise ValueError("Expected a checkpoint with the learned second depth layer.")
    rgb, _, inferred_focal = load_rgb(args.input)
    focal_px = float(inferred_focal if args.focal_px is None else args.focal_px)
    if not math.isfinite(focal_px) or focal_px <= 0:
        raise ValueError("The focal length must be positive and finite.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    da3 = FrozenDA3(args.da3_checkpoint, device)
    features = da3.extract(rgb, focal_px, focal_px)
    del da3
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = FullDA3Predictor(args.sharp_checkpoint)
    model.load_trainable_state_dict(state["trainable"])
    del state
    scene = reconstruct(model.to(device).eval(), rgb, features, focal_px, args.resolution, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        pending = Path(temp) / "scene.ply"
        save_ply(scene, focal_px, rgb.shape[:2], pending)
        pending.replace(args.output)
    print(
        f"Saved {scene.mean_vectors.shape[1]} Gaussians (focal {focal_px:.2f} px) to {args.output}"
    )


def main():
    """Parse inference paths and the optional source-camera override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Single source RGB photograph.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .ply file.")
    parser.add_argument(
        "--da3-checkpoint", type=Path, required=True, help="DA3METRIC-LARGE directory."
    )
    parser.add_argument(
        "--sharp-checkpoint", type=Path, required=True, help="Official SHARP .pt file."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "learned-tuned.pt",
        help="Experiment weights. Defaults to the bundled fine-tuned Gaussian decoder.",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        help="Focal length in native-image pixels. Defaults to SHARP's EXIF estimate.",
    )
    parser.add_argument(
        "--resolution", type=int, default=1536, help="Gaussian decoder resolution, divisible by 32."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Default: CUDA when available, otherwise CPU.",
    )
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
