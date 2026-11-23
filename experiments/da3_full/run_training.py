"""Run the synthetic and WildRGB-D training stages for the two Gaussian decoders."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run(script, *arguments):
    """Use a fresh process for each stage to release CUDA memory between models."""
    command = [sys.executable, "-u", str(SCRIPTS / script), *map(str, arguments)]
    subprocess.run(command, check=True)


def main(args):
    """Prepare the datasets, train the teacher, and fine-tune both decoder configurations."""
    root = args.output.resolve()
    real, synthetic = root / "wildrgbd", root / "synthetic"
    real_manifest, synthetic_manifest = real / "manifest.json", synthetic / "manifest.json"
    if not synthetic_manifest.exists():
        run("synthetic_data.py", "--output", synthetic)
    if not real_manifest.exists():
        run("prepare_categories.py", "--archives", args.archives, "--output", real)

    def cache(manifest, directory):
        run(
            "cache_features.py",
            "--manifest",
            manifest,
            "--cache",
            directory,
            "--da3-checkpoint",
            args.da3_checkpoint,
        )

    def train(output, stage, steps, manifest, features, extra=()):
        resume = ["--resume"] if (output / "last.pt").exists() else []
        run(
            "train.py",
            "--stage",
            stage,
            "--output",
            output,
            "--steps",
            steps,
            "--manifest",
            manifest,
            "--cache",
            features,
            "--sharp-checkpoint",
            args.sharp_checkpoint,
            "--vgg-weights",
            args.vgg_weights,
            "--resolution",
            768,
            "--lr",
            0.00016,
            "--warmup",
            round(0.04 * steps),
            "--validate-every",
            160 if stage == 1 else 96,
            "--val-pairs",
            32 if stage == 1 else 24,
            "--save-every",
            40,
            "--log-every",
            20,
            "--grad-clip",
            10,
            "--seed",
            617,
            *extra,
            *resume,
        )

    synthetic_cache = root / "synthetic-features"
    real_cache = root / "wild-features"
    pseudo_cache = root / "pseudo-features"
    cache(synthetic_manifest, synthetic_cache)
    stage1 = root / "stage1"
    train(stage1, 1, 960, synthetic_manifest, synthetic_cache)
    teacher = stage1 / "best.pt"
    cache(real_manifest, real_cache)
    pseudo_manifest = real / "pseudo.json"
    run(
        "self_supervision.py",
        "--manifest",
        real_manifest,
        "--cache",
        real_cache,
        "--checkpoint",
        teacher,
        "--sharp-checkpoint",
        args.sharp_checkpoint,
        "--output",
        pseudo_manifest,
        "--resolution",
        768,
    )
    cache(pseudo_manifest, pseudo_cache)
    common = [
        "--previous-stage",
        teacher,
        "--validation-manifest",
        real_manifest,
        "--validation-cache",
        real_cache,
    ]
    train(root / "frozen", 2, 288, pseudo_manifest, pseudo_cache, common)
    train(
        root / "tuned",
        2,
        288,
        pseudo_manifest,
        pseudo_cache,
        [*common, "--tune-decoder-tail", "--decoder-lr-factor", 0.1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--archives", required=True, type=Path, help="WildRGB-D ZIP archives or download cache"
    )
    parser.add_argument("--sharp-checkpoint", required=True, type=Path)
    parser.add_argument("--da3-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--vgg-weights", required=True, type=Path, help="Torchvision VGG16 ImageNet weights"
    )
    main(parser.parse_args())
