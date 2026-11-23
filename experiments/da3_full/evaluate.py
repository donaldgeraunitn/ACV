"""Evaluate the learned DA3 frozen- and tuned-decoder checkpoints on Middlebury 2014."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from metrics import dists_full_frame, verify_dists
from PIL import Image

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "results" / "middlebury-inputs.json"
ARMS = ("frozen", "tuned")


def sha(path):
    """Hash a checkpoint or evaluation specification."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Replace a completed result atomically."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def cameras(folder):
    """Read native rectified intrinsics, metric camera baseline and dimensions."""
    cfg = dict(line.split("=", 1) for line in (folder / "calib.txt").read_text().splitlines())
    ks = [
        np.fromstring(cfg[k].strip("[]").replace(";", " "), sep=" ").reshape(3, 3)
        for k in ("cam0", "cam1")
    ]
    return (
        ks[0],
        ks[1],
        float(cfg["baseline"]) / 1000,
        int(cfg["height"]),
        int(cfg["width"]),
    )


def benchmark_folders(root):
    """Validate the two photos and rectified calibration used for each reference scene."""
    expected = json.loads(INPUTS.read_text())
    for scene, files in expected.items():
        for filename, digest in files.items():
            path = root / scene / filename
            if sha(path) != digest:
                raise ValueError(f"Unexpected Middlebury file contents: {path}")
    return [root / scene for scene in sorted(expected)]


def read_checkpoint(path, arm):
    """Require the stage-two learned depth head and corresponding decoder weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["stage"], state["step"], state["variant"]) != (2, 288, "learned"):
        raise ValueError(f"Expected the learned checkpoint at update 288: {path}")
    weights = state["trainable"]
    if "second_head" not in weights or ("decoder_tail" in weights) != (arm == "tuned"):
        raise ValueError(f"Expected the {arm} Gaussian decoder: {path}")
    return weights


def extract_features(args, scenes, encoded):
    """Compute missing DA3 source features once in the current evaluation directory."""
    from model import FrozenDA3

    encoded.mkdir(exist_ok=True)
    backbone = None
    for folder in scenes:
        path = encoded / (folder.name + ".pt")
        if path.exists():
            continue
        if backbone is None:
            backbone = FrozenDA3(args.da3_checkpoint, "cuda", process_res=504)
        k0, _, _, _, _ = cameras(folder)
        with Image.open(folder / "im0.png") as photo:
            rgb = np.array(photo.convert("RGB"))
        data = backbone.extract(rgb, float(k0[0, 0]), float(k0[1, 1]))
        tmp = path.with_suffix(".tmp")
        torch.save(data, tmp)
        tmp.replace(path)


def evaluate_render(args, scenes):
    """Render both arms with the official renderer and identical calibration."""
    from model import FullDA3Predictor
    from sharp.utils.gaussians import unproject_gaussians
    from sharp.utils.gsplat import GSplatRenderer

    def k4(k):
        kk = torch.eye(4, device="cuda")
        kk[:3, :3] = torch.as_tensor(k, device="cuda", dtype=torch.float32)
        return kk

    paths = {arm: getattr(args, arm) for arm in ARMS}
    weights = {arm: read_checkpoint(path, arm) for arm, path in paths.items()}
    spec = {
        "checkpoints": {a: sha(p) for a, p in paths.items()},
        "sharp_sha": sha(args.sharp_checkpoint),
        "da3_sha": sha(args.da3_checkpoint / "model.safetensors"),
        "resolution": args.resolution,
        "process_res": 504,
        "benchmark_sha": sha(INPUTS),
    }
    encoded = args.output / "source-features"
    spec_path = args.output / "evaluation-spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Choose a different output directory for these checkpoints or resolution")
    write_json(spec_path, spec)
    extract_features(args, scenes, encoded)
    torch.cuda.empty_cache()
    predictors = {}
    for arm in ARMS:
        net = FullDA3Predictor(args.sharp_checkpoint)
        net.load_trainable_state_dict(weights.pop(arm))
        predictors[arm] = net.to("cuda").eval()
    renderer = GSplatRenderer(color_space="linearRGB").eval()
    device = "cuda"
    for folder in scenes:
        scene = folder.name.removesuffix("-perfect")
        dest = args.output / "renders" / scene
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "done.json").exists() and all((dest / f"{arm}.npy").exists() for arm in ARMS):
            continue
        k0, k1, baseline, h, w = cameras(folder)
        with Image.open(folder / "im0.png") as image:
            rgb = np.array(image.convert("RGB"))
        assert rgb.shape == (h, w, 3)
        data = torch.load(encoded / (folder.name + ".pt"), weights_only=True, map_location="cpu")
        dep = data["depth"].to(device=device, dtype=torch.float32)
        fs = [x.to(device=device, dtype=torch.float32) for x in data["features"]]
        tensor = torch.from_numpy(rgb).to(device).float().permute(2, 0, 1)[None] / 255
        n = args.resolution
        image = F.interpolate(tensor, size=(n, n), mode="bilinear", align_corners=True)
        ks = k4(k0)
        ks[0] *= n / w
        ks[1] *= n / h
        kt = k4(k1)[None]
        pose = torch.eye(4, device="cuda")
        pose[0, 3] = -baseline
        diagnostics = {}
        for arm, model in predictors.items():
            ndc, layers = model(image, dep, fs)
            signed = (layers[:, 1] / layers[:, 0]).log()
            diagnostics[arm] = {
                "median_abs_log_depth_ratio": float(signed.abs().median()),
                "gaussians": ndc.mean_vectors.shape[1],
            }
            world = unproject_gaussians(ndc, torch.eye(4, device=device), ks, (n, n))
            result = renderer(world, pose[None], kt, w, h)
            pixels = result.color[0].permute(1, 2, 0).clamp(0, 1).contiguous().cpu().numpy()
            assert np.isfinite(pixels).all()
            tmp = dest / f"{arm}.tmp.npy"
            np.save(tmp, pixels)
            tmp.replace(dest / f"{arm}.npy")
            del result, world, ndc, layers, pixels
        write_json(dest / "done.json", diagnostics)
        print("Rendered", scene, diagnostics, flush=True)


def evaluate_scores(args, scenes):
    """Score the held-out right views and aggregate scene-weighted results."""
    import lpips
    from DISTS_pytorch import DISTS

    with args.original_scores.open(newline="") as stream:
        originals = {r["scene"]: r for r in csv.DictReader(stream)}
    if set(originals) != {s.name.removesuffix("-perfect") for s in scenes}:
        raise ValueError("SHARP reference scores must cover the 23 benchmark scenes")
    spec_path = args.output / "evaluation-spec.json"
    specification = json.loads(spec_path.read_text())
    if set(specification["checkpoints"]) != set(ARMS) or specification["benchmark_sha"] != sha(
        INPUTS
    ):
        raise ValueError(f"Unexpected render specification: {spec_path}")
    spec = {"render": sha(spec_path), "sharp_reference": sha(args.original_scores)}
    ds = DISTS().eval().to("cuda")
    lp = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval().to("cuda")
    rows = []
    (args.output / "scores").mkdir(exist_ok=True)
    for folder in scenes:
        scene = folder.name.removesuffix("-perfect")
        f = args.output / "scores" / (scene + ".json")
        if f.exists():
            saved = json.loads(f.read_text())
            if saved["spec"] == spec:
                rows.append(saved["row"])
                continue
        with Image.open(folder / "im1.png") as im:
            gt = np.array(im.convert("RGB"), dtype=np.float32) / 255
        renders = args.output / "renders" / scene
        arrays = [gt] + [np.load(renders / (arm + ".npy"), mmap_mode="c") for arm in ARMS]
        assert all(a.shape == gt.shape and a.dtype == np.float32 for a in arrays)
        images = torch.stack([torch.from_numpy(x).permute(2, 0, 1) for x in arrays]).to("cuda")
        if scene == "Adirondack":
            write_json(
                args.output / "dists-validation.json",
                verify_dists(ds, images[:, :, :590, :810]),
            )
        lv = [lp(images[:1], images[i : i + 1], normalize=True).item() for i in (1, 2)]
        dv = dists_full_frame(images, ds)
        row = {
            "scene": scene,
            **{f"sharp_{m}": float(originals[scene][f"sharp_{m}"]) for m in ("lpips", "dists")},
        }
        for arm, v1, v2 in zip(ARMS, lv, dv, strict=True):
            row[arm + "_lpips"] = float(v1)
            row[arm + "_dists"] = float(v2)
        assert all(np.isfinite(v) for k, v in row.items() if k != "scene")
        write_json(f, {"spec": spec, "row": row})
        rows.append(row)
        print("Scored", scene, row, flush=True)
        del images
    output = args.output / "per-scene.csv"
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)
    write_json(args.output / "summary.json", summarize(rows))


def summarize(rows):
    """Return equal-scene means and paired bootstrap intervals for tuned minus frozen."""
    summary = {
        "scenes": len(rows),
        "aggregation": "Equal scene weights, native full right view",
    }
    samples = np.random.default_rng(617).integers(0, len(rows), size=(20_000, len(rows)))
    for metric in ("lpips", "dists"):
        means = {a: float(np.mean([r[f"{a}_{metric}"] for r in rows])) for a in ("sharp", *ARMS)}
        deltas = np.array([r[f"tuned_{metric}"] - r[f"frozen_{metric}"] for r in rows])
        means["tuned_better_than_frozen"] = int((deltas < 0).sum())
        means["tuned_minus_frozen_mean"] = float(deltas.mean())
        means["paired_scene_bootstrap_95pct"] = np.quantile(
            deltas[samples].mean(axis=1), [0.025, 0.975]
        ).tolist()
        summary[metric] = means
    return summary


def main():
    """Run either model inference or perceptual scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    render = actions.add_parser("render", parents=[common], help="Render calibrated right views")
    render.add_argument("--sharp-checkpoint", type=Path, required=True)
    render.add_argument("--da3-checkpoint", type=Path, required=True)
    for arm in ARMS:
        render.add_argument(
            "--" + arm, type=Path, default=HERE / "checkpoints" / f"learned-{arm}.pt"
        )
    score = actions.add_parser("score", parents=[common], help="Compute full-frame LPIPS and DISTS")
    score.add_argument("--original-scores", type=Path, default=HERE / "results" / "middlebury.csv")
    args = parser.parse_args()
    args.resolution = 1536
    scenes = benchmark_folders(args.dataset)
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        if args.action == "render":
            evaluate_render(args, scenes)
        else:
            evaluate_scores(args, scenes)


if __name__ == "__main__":
    main()
