"""Train the DA3 feature adaptor, learned second layer and optional Gaussian decoder tail."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import DECODER_TAIL_MODULES, FullDA3Predictor
from PIL import Image
from render import frustum_mask, render_views
from torch import nn
from torchvision.models import vgg


def read_manifest(path):
    """Read camera pairs and keep their image paths relative to the manifest."""
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    return [dict(row, root=path.parent) for row in pairs]


def rgb_frame(path, resolution, device):
    """Return resampled sRGB values and the native image size used by the calibration."""
    with Image.open(path) as im:
        array = np.array(im.convert("RGB"))
    tensor = (
        torch.from_numpy(array).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255
    )
    size = array.shape[:2]
    return F.interpolate(
        tensor, size=(resolution, resolution), mode="bilinear", align_corners=False, antialias=True
    ), size


def depth_frame(row, field, resolution, device):
    """Read optional metric camera-space depth using nearest-neighbor sampling."""
    if not row.get(field):
        return None
    path = row["root"] / row[field]
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)  # metric z, metres
    else:
        with Image.open(path) as im:
            array = np.array(im, dtype=np.float32) * 0.001  # uint16 millimetres
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D z-depth image at {path}")
    depth = torch.from_numpy(array).to(device)[None, None]
    return F.interpolate(depth, size=(resolution, resolution), mode="nearest-exact")


def scaled_camera(native, size, res, device):
    """Scale native intrinsics to the square training image."""
    h, w = size
    K = torch.tensor(native, dtype=torch.float32, device=device)
    K[0] *= res / w
    K[1] *= res / h
    return K


def load_pair(row, cache_root, args):
    """Load two photographs, calibration, optional depths and frozen source features."""
    src, src_hw = rgb_frame(row["root"] / row["source"], args.resolution, args.device)
    tgt, tgt_hw = rgb_frame(row["root"] / row["target"], args.resolution, args.device)
    key = Path(row.get("source_cache", Path(row["source"]).with_suffix(".pt")))
    frozen = torch.load(cache_root / key, map_location="cpu", weights_only=True)
    Ks = scaled_camera(row["K_source"], src_hw, args.resolution, args.device)
    Kt = scaled_camera(row["K_target"], tgt_hw, args.resolution, args.device)
    pose = torch.tensor(row["target_from_source"], dtype=torch.float32, device=args.device)
    sd, td = (
        depth_frame(row, k, args.resolution, args.device) for k in ("source_depth", "target_depth")
    )
    return src, tgt, frozen, Ks, Kt, pose, sd, td


class VGGPerceptual(nn.Module):
    """Four fixed ImageNet feature blocks and spatially normalized Gram matrices."""

    def __init__(self, path):
        """Load the four pretrained VGG blocks used by the perceptual loss."""
        super().__init__()
        self.features = vgg.make_layers(vgg.cfgs["D"])[:23]
        weights = torch.load(path, map_location="cpu", weights_only=True, mmap=False)
        state = {k: weights["features." + k] for k in self.features.state_dict()}
        self.features.load_state_dict(state, strict=True)
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def maps(self, pixels):
        """Extract the final activation of each of the four feature blocks."""
        features = (pixels - self.mean) / self.std
        maps = []
        for i, layer in enumerate(self.features):
            features = layer(features)
            if i in (3, 8, 15, 22):
                maps.append(features)
        return maps

    @staticmethod
    def gram(features):
        """Return channel correlations averaged over spatial locations."""
        f = features.flatten(2)
        return f @ f.transpose(1, 2) / f.shape[-1]

    def forward(self, pred, target, mask):
        """Compare the feature maps and Gram matrices on supported target pixels."""
        # Replacing unsupported pixels by target values cuts their gradients while
        # preserving the natural target context for feature receptive fields.
        pred = torch.where(mask, pred, target)
        with torch.no_grad():
            target_maps = self.maps(target)
        loss = pred.new_zeros(())
        for x, y in zip(self.maps(pred), target_maps, strict=True):
            loss = loss + F.mse_loss(x, y) + 10 * F.mse_loss(self.gram(x), self.gram(y))
        return loss


def masked_mean(values, mask):
    """Average selected values, counting each valid channel separately."""
    mask = mask.expand_as(values)
    return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1)


def gaussian_regularizers(gaussians, depths):
    """Compute inverse-depth smoothness, edge opacity and lateral offset penalties."""
    inv = depths.clamp_min(1e-5).reciprocal()
    dx = F.pad(inv[..., 1:] - inv[..., :-1], (0, 1, 0, 0))
    dy = F.pad(inv[..., 1:, :] - inv[..., :-1, :], (0, 0, 0, 1))
    tv = dx[:, 1].abs().mean() + dy[:, 1].abs().mean()
    # Gaussian centers start at the center of each 2x2 initializer cell.
    magnitude = F.avg_pool2d((dx.square() + dy.square() + 1e-12).sqrt(), 2)
    grad = (
        gaussians.opacities.reshape_as(magnitude)
        * (-torch.expm1(-((magnitude - 0.01).relu() / 0.01)))
    ).mean()
    _, layers, h, w = magnitude.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depths.device), torch.arange(w, device=depths.device), indexing="ij"
    )
    xx, yy = (xx + 0.5) * 2 / w - 1, (yy + 0.5) * 2 / h - 1
    p = gaussians.mean_vectors.reshape(1, layers, h, w, 3)
    x, y, z = p.unbind(-1)
    # The original composer applies a 0.001 multiplier to the raw xy deltas.
    xy = (
        ((x / z - xx).abs() / 0.001 - 400).relu() + ((y / z - yy).abs() / 0.001 - 400).relu()
    ).mean()
    return tv, grad, xy


def objective(row, cache, args, model, perceptual):
    """Render a pair and compute the training loss and its component measurements."""
    source, target, frozen, Ks, Kt, pose, sd, td = load_pair(row, cache, args)
    gs, depths = model(source, frozen["depth"], frozen["features"])
    size = tuple(source.shape[-2:])
    ident = torch.eye(4, device=source.device)
    out = render_views(
        gs,
        Ks,
        size,
        torch.stack([Ks, Kt]),
        torch.stack([ident, pose]),
        size,
        compute_splat_penalty=True,
    )
    if td is None:
        mask = torch.ones_like(target[:, :1], dtype=torch.bool)
    else:
        mask = frustum_mask(td, Kt, Ks, pose, size)
        if not bool(mask.any()):
            raise ValueError(f"No valid target depth/overlap for {row['scene']} {row['target']}")
    color = (out.rgb[:1] - source).abs().mean() + masked_mean((out.rgb[1:] - target).abs(), mask)
    alpha = -torch.log(out.alpha[:1].clamp(1e-6, 1)).mean() - masked_mean(
        torch.log(out.alpha[1:].clamp(1e-6, 1)), mask
    )
    with torch.no_grad():
        # DA3 is frozen, so this first-layer depth term carries no gradient.
        depth = color.new_zeros(())
        if sd is not None:
            valid = (sd > 0) & torch.isfinite(sd)
            depth = masked_mean(
                (
                    depths[:, :1].clamp_min(1e-5).reciprocal() - sd.clamp_min(1e-5).reciprocal()
                ).abs(),
                valid,
            )
    percept = perceptual(out.rgb[1:], target, mask)
    tv, grad, xy = gaussian_regularizers(gs, depths)
    total = color + alpha + 3 * percept + 0.2 * depth + tv + 0.5 * grad + xy + out.splat_penalty
    metrics = {
        "total": total,
        "novel_l1": masked_mean((out.rgb[1:] - target).abs(), mask),
        "color": color,
        "percept": percept,
        "alpha": alpha,
        "frozen_depth": depth,
        "tv": tv,
        "grad": grad,
        "xy": xy,
        "splat": out.splat_penalty,
        "valid_target": mask.float().mean(),
        "abs_second_log_ratio": (depths[:, 1] / depths[:, 0]).log().abs().mean(),
    }
    return total, metrics


def atomic_save(state, path):
    """Replace a checkpoint only after its new state has finished writing."""
    temp = path.with_suffix(".partial")
    torch.save(state, temp)
    temp.replace(path)


def log(args, record):
    """Append one training or validation event to disk and report it to the console."""
    with (args.output / "history.jsonl").open("a", encoding="utf-8") as out:
        print(json.dumps(record, allow_nan=False), file=out)
    print(json.dumps(record, allow_nan=False), flush=True)


def validate(rows, cache, args, model, perceptual, step):
    """Evaluate fixed validation views and return their mean target-view L1 error."""
    model.eval()
    totals = {}
    with torch.no_grad():
        for row in rows:
            _, metrics = objective(row, cache, args, model, perceptual)
            for key, tensor in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(tensor)
    result = {key: x / len(rows) for key, x in totals.items()}
    log(args, {"event": "val", "step": step, "n": len(rows), **result})
    return result["novel_l1"]


def lr_at(step, args):
    """Keep the final learning rate after the original warmup and cosine schedule ends."""
    step = min(step, args.steps)
    if step <= args.warmup:
        return args.lr * step / args.warmup
    t = (step - args.warmup) / max(args.steps - args.warmup, 1)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(t * math.pi)))


def main(args):
    """Train or resume one stage of the two-stage curriculum."""
    args.output.mkdir(parents=True, exist_ok=True)
    name = args.output / "last.pt"
    history = args.output / "history.jsonl"
    if (name.exists() or history.exists()) and not args.resume:
        raise FileExistsError(f"A run exists at {args.output}. Choose a new run or use --resume.")
    if args.stage == 2 and not args.previous_stage:
        raise ValueError("Stage 2 requires the corresponding stage-1 model checkpoint.")
    if args.stage == 2:
        pseudo = json.loads(args.manifest.read_text(encoding="utf-8"))
        with args.previous_stage.open("rb") as f:
            teacher_hash = hashlib.file_digest(f, "sha256").hexdigest()
        if (
            pseudo.get("stage") != 2
            or pseudo.get("variant") != "learned"
            or pseudo.get("teacher_sha256") != teacher_hash
        ):
            raise ValueError("Stage-2 data must come from this variant's exact stage-1 teacher.")
    if (
        not 0 <= args.warmup <= args.steps
        or args.steps < 1
        or args.resolution < 32
        or args.resolution % 32
    ):
        raise ValueError(
            "Positive steps, warmup in [0,steps], and resolution divisible by 32 required."
        )
    if min(args.save_every, args.validate_every, args.log_every, args.val_pairs) <= 0:
        raise ValueError(
            "Logging/checkpoint/validation intervals and validation pair count must be positive."
        )
    if not 0 < args.decoder_lr_factor <= 1:
        raise ValueError("Expected a decoder LR factor in (0, 1].")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    rows = read_manifest(args.manifest)
    train = [r for r in rows if r["split"] == "train"]
    val = [
        r
        for r in read_manifest(args.validation_manifest or args.manifest)
        if r["split"] in ("val", "validation")
    ]
    if not train or not val:
        raise ValueError("Training and validation must both include nonempty scene-disjoint pairs.")
    if {r["scene"] for r in train} & {r["scene"] for r in val}:
        raise ValueError("Train and validation scenes overlap.")
    # Pick validation rows across scenes, keeping one row per scene first.
    groups = {}
    for row in val:
        groups.setdefault(row["scene"], []).append(row)
    val = [
        group[i]
        for i in range(max(map(len, groups.values())))
        for group in groups.values()
        if i < len(group)
    ][: args.val_pairs]
    schedule = []
    rng = random.Random(args.seed)
    while len(schedule) < args.steps:
        epoch = list(range(len(train)))
        rng.shuffle(epoch)
        schedule.extend(epoch)

    model = FullDA3Predictor(
        args.sharp_checkpoint,
        tune_decoder_tail=args.tune_decoder_tail,
    ).to(args.device)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not params:
        raise RuntimeError("No trainable parameters.")
    perceptual = VGGPerceptual(args.vgg_weights).eval().to(args.device)
    tail_prefixes = tuple(name + "." for name in DECODER_TAIL_MODULES)
    adapter_params = [p for n, p in params.items() if not n.startswith(tail_prefixes)]
    optimizer = torch.optim.Adam(adapter_params, lr=args.lr)
    if args.tune_decoder_tail:
        tail_params = [p for n, p in params.items() if n.startswith(tail_prefixes)]
        optimizer.add_param_group({"params": tail_params, "lr": args.lr * args.decoder_lr_factor})
    start, best = 0, math.inf
    config = {
        k: (str(v.resolve()) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k != "resume"
    }
    config["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    config["validation_manifest_sha256"] = hashlib.sha256(
        (args.validation_manifest or args.manifest).read_bytes()
    ).hexdigest()
    if args.resume or args.previous_stage:
        state = torch.load(
            name if args.resume else args.previous_stage, weights_only=True, map_location="cpu"
        )
        if state["variant"] != "learned" or state["stage"] != (
            args.stage if args.resume else args.stage - 1
        ):
            raise ValueError("Checkpoint stage or variant mismatch.")
        model.load_trainable_state_dict(
            state["trainable"], initialize_from_frozen=args.tune_decoder_tail and not args.resume
        )
        if args.resume:
            if state["config"] != config:
                raise ValueError("Training options differ from the saved run.")
            optimizer.load_state_dict(state["optimizer"])
            start, best = state["step"], state["best_val_novel_l1"]
    if start == 0:
        log(
            args,
            {
                "event": "config",
                **config,
                "trainable_names": list(params),
                "parameter_count": sum(p.numel() for p in params.values()),
                "train_pairs": len(train),
                "val_scenes": sorted(set(r["scene"] for r in val)),
                "perceptual_network": "VGG16-ImageNet",
                "depth_alignment": "identity",
            },
        )
        validate(val, args.validation_cache or args.cache, args, model, perceptual, 0)
    for idx in range(start, args.steps):
        step = idx + 1
        row = train[schedule[idx]]
        lr = lr_at(step, args)
        optimizer.param_groups[0]["lr"] = lr
        if args.tune_decoder_tail:
            optimizer.param_groups[1]["lr"] = lr * args.decoder_lr_factor
        # Frozen SHARP modules remain in evaluation mode, keeping batchnorm/dropout fixed.
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(row, args.cache, args, model, perceptual)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training objective at step {step} for {row['source']}")
        loss.backward()
        norms = {k: p.grad.detach().norm() for k, p in params.items() if p.grad is not None}
        if not norms or not torch.stack(list(norms.values())).isfinite().all():
            raise RuntimeError(f"Invalid or missing gradients at step {step} for {row['source']}")
        # Preserve the control's clipping budget for the adapter and second depth head.
        adapter_norm = torch.nn.utils.clip_grad_norm_(
            adapter_params, args.grad_clip, error_if_nonfinite=True
        )
        if idx == start and not adapter_norm.item() > 0:
            raise RuntimeError("Gaussian decoder has no gradient to the DA3 trainable modules.")
        if args.tune_decoder_tail:
            decoder_norm = torch.nn.utils.clip_grad_norm_(
                tail_params, args.grad_clip, error_if_nonfinite=True
            )
            if idx == start:
                for prefix in tail_prefixes:
                    group_norms = [v for n, v in norms.items() if n.startswith(prefix)]
                    if not group_norms or not torch.stack(group_norms).max().item() > 0:
                        raise RuntimeError(f"No gradient reached decoder module {prefix[:-1]}.")
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "event": "train",
                "step": step,
                "scene": row["scene"],
                "source": row["source"],
                "grad_norm": float(adapter_norm),
                "lr": optimizer.param_groups[0]["lr"],
                **{k: float(v.detach()) for k, v in metrics.items()},
            }
            if args.tune_decoder_tail:
                record["decoder_grad_norm"] = float(decoder_norm)
                record["decoder_lr"] = optimizer.param_groups[1]["lr"]
            if step == 1:
                record["gradient_norms"] = {k: float(v) for k, v in norms.items()}
            log(args, record)
        del loss, metrics, norms
        improved = False
        if step % args.validate_every == 0 or step == args.steps:
            val_score = validate(
                val, args.validation_cache or args.cache, args, model, perceptual, step
            )
            improved = val_score < best
            best = min(best, val_score)
        if improved or step == 1 or step % args.save_every == 0 or step == args.steps:
            state = {
                "stage": args.stage,
                "step": step,
                "variant": "learned",
                "config": config,
                "trainable": model.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_novel_l1": best,
            }
            atomic_save(state, name)
            if improved:
                atomic_save(state, args.output / "best.pt")


def parse_args():
    """Parse dataset paths, stage and optimization settings."""
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "cache", "output", "sharp-checkpoint", "vgg-weights"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--validation-manifest", type=Path)
    p.add_argument("--validation-cache", type=Path)
    p.add_argument("--previous-stage", type=Path)
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--lr", type=float, default=1.6e-4)
    p.add_argument("--tune-decoder-tail", action="store_true")
    p.add_argument("--decoder-lr-factor", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--validate-every", type=int, default=25)
    p.add_argument("--val-pairs", type=int, default=4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=617)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
