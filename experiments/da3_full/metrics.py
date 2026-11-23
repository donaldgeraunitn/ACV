"""Full-frame DISTS for two predictions on a memory-limited GPU."""

import math

import numpy as np
import torch


def dists_full_frame(images, metric, core=768):
    """Evaluate official DISTS features with exact full-frame spatial moments.

    All stages are convolutional. A stage-5 feature uses at most 105 input pixels
    of context on each side. Tile origins and halos are aligned to the final 16x
    stride. Only tile interiors contribute, including the true outer image border.
    """
    assert core > 0 and core % 16 == 0
    halo = 112
    _, _, height, width = images.shape
    strides = (1, 1, 2, 4, 8, 16)
    stages = (metric.stage1, metric.stage2, metric.stage3, metric.stage4, metric.stage5)
    # Per channel: sum and square-sum for the target and two predictions, plus two cross sums.
    sums = [torch.zeros((8, c), device=images.device, dtype=torch.float64) for c in metric.chns]
    counts = [0] * len(strides)
    for y in range(0, height, core):
        top, bottom = max(0, y - halo), min(height, y + core + halo)
        stop_y = min(height, y + core)
        for x in range(0, width, core):
            left, right = max(0, x - halo), min(width, x + core + halo)
            stop_x = min(width, x + core)
            h = images[:, :, top:bottom, left:right].contiguous()
            for i, stride in enumerate(strides):
                if i == 1:
                    h = (h - metric.mean) / metric.std
                if i:
                    h = stages[i - 1](h)
                interior = h[
                    :,
                    :,
                    (y - top) // stride : math.ceil((stop_y - top) / stride),
                    (x - left) // stride : math.ceil((stop_x - left) / stride),
                ]
                counts[i] += interior.shape[-2] * interior.shape[-1]
                sums[i][:3] += interior.sum(dim=(-2, -1), dtype=torch.float64)
                sums[i][3:6] += interior.square().sum(dim=(-2, -1), dtype=torch.float64)
                sums[i][6:] += (interior[:1] * interior[1:]).sum(dim=(-2, -1), dtype=torch.float64)
                del interior

    assert counts == [math.ceil(height / stride) * math.ceil(width / stride) for stride in strides]
    normalizer = metric.alpha.sum() + metric.beta.sum()
    alphas = (metric.alpha.flatten() / normalizer).split(metric.chns)
    betas = (metric.beta.flatten() / normalizer).split(metric.chns)
    similarity = torch.zeros(2, dtype=torch.float64, device=images.device)
    for moments, n, alpha, beta in zip(sums, counts, alphas, betas, strict=True):
        averages = moments / n
        mean = averages[:3]
        variance = (averages[3:6] - mean.square()).clamp_min(0)
        covariance = averages[6:] - mean[:1] * mean[1:]
        mean_term = (2 * mean[:1] * mean[1:] + 1e-6) / (
            mean[:1].square() + mean[1:].square() + 1e-6
        )
        covariance_term = (2 * covariance + 1e-6) / (variance[:1] + variance[1:] + 1e-6)
        similarity += (alpha * mean_term + beta * covariance_term).sum(dim=1)
    return (1 - similarity).cpu().numpy()


def verify_dists(metric, images):
    """Check tile boundary handling against official DISTS on a full-resolution crop."""
    crop = images[:, :, 113:590, 157:810].contiguous()
    assert crop.shape[-2:] == (477, 653)
    tiled = dists_full_frame(crop, metric, core=224)
    direct = np.array([metric(crop[:1], crop[i : i + 1]).item() for i in (1, 2)])
    difference = np.abs(direct - tiled)
    np.testing.assert_allclose(tiled, direct, rtol=0, atol=2e-5)
    return {
        "scene": "Adirondack",
        "crop_xywh": [157, 113, 653, 477],
        "tile_core": 224,
        "reference": direct.tolist(),
        "full_frame_moments": tiled.tolist(),
        "max_abs_difference": float(difference.max()),
    }
