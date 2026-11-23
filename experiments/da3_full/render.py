"""Differentiable camera projection for SHARP's metric NDC Gaussians."""

from typing import NamedTuple

import gsplat
import torch
from sharp.utils.color_space import linearRGB2sRGB
from sharp.utils.gaussians import Gaussians3D, compose_covariance_matrices


class Views(NamedTuple):
    """Images, coverage, metric depths and optional scale penalty for rendered camera views."""

    rgb: torch.Tensor  # [V, 3, H, W], sRGB
    alpha: torch.Tensor  # [V, 1, H, W]
    depth: torch.Tensor  # [V, 1, H, W], expected camera-space z in metres
    splat_penalty: torch.Tensor


def camera_gaussians(gaussians: Gaussians3D, intrinsics: torch.Tensor, source_hw: tuple[int, int]):
    """Transform full covariances by congruence, retaining their autograd graph."""
    h, w = source_hw
    pixel_from_ndc = intrinsics.new_tensor([[w / 2, 0, w / 2], [0, h / 2, h / 2], [0, 0, 1]])
    source_from_ndc = torch.linalg.solve(intrinsics, pixel_from_ndc)
    means = gaussians.mean_vectors[0] @ source_from_ndc.T
    covariance_ndc = compose_covariance_matrices(
        gaussians.quaternions[0], gaussians.singular_values[0]
    )
    covariance = source_from_ndc @ covariance_ndc @ source_from_ndc.T
    return means, covariance


def render_views(
    gaussians: Gaussians3D,
    source_intrinsics: torch.Tensor,
    source_hw: tuple[int, int],
    view_intrinsics: torch.Tensor,
    view_from_source: torch.Tensor,
    out_hw: tuple[int, int],
    compute_splat_penalty: bool = False,
) -> Views:
    """Render calibrated OpenCV views. `source_intrinsics` refers to `source_hw`."""
    if gaussians.mean_vectors.shape[0] != 1:
        raise ValueError("Supply one Gaussian scene at a time.")
    means, covariance = camera_gaussians(gaussians, source_intrinsics, source_hw)
    h, w = out_hw
    channels, alpha, meta = gsplat.rendering.rasterization(
        means=means.contiguous(),
        covars=covariance.contiguous(),
        quats=None,
        scales=None,
        opacities=gaussians.opacities[0].contiguous(),
        colors=gaussians.colors[0].contiguous(),
        viewmats=view_from_source.contiguous(),
        Ks=view_intrinsics.contiguous(),
        width=w,
        height=h,
        packed=False,
        render_mode="RGB+D",
        rasterize_mode="classic",
        eps2d=0.0,
    )
    rgb = linearRGB2sRGB(channels[..., :3].permute(0, 3, 1, 2))
    a = alpha.permute(0, 3, 1, 2)
    depth = channels[..., 3:4].permute(0, 3, 1, 2) / a.clamp_min(1e-8)
    splat = means.new_zeros(())
    if compute_splat_penalty:
        # The conics are inverse pixel covariances. Penalize standard deviations
        # outside [0.1, 100] as in SHARP Eq. 3.10. Culled splats have no conic.
        a11, a12, a22 = meta["conics"].unbind(-1)
        visible = (meta["radii"] > 0).all(-1)
        a11 = torch.where(visible, a11, 1.0)
        a12 = torch.where(visible, a12, 0.0)
        a22 = torch.where(visible, a22, 1.0)
        trace = a11 + a22
        difference = ((a11 - a22).square() + 4 * a12.square() + 1e-12).sqrt()
        eig_high = ((trace + difference) / 2).clamp_min(1e-12)
        eig_low = ((a11 * a22 - a12.square()) / eig_high).clamp_min(1e-12)
        deviations = (eig_low.rsqrt() - 100).relu() + (0.1 - eig_high.rsqrt()).relu()
        splat = torch.where(visible, deviations, 0).sum() / visible.sum().clamp_min(1)
    return Views(rgb, a, depth, splat)


def frustum_mask(
    depth: torch.Tensor,
    target_K: torch.Tensor,
    source_K: torch.Tensor,
    target_from_source: torch.Tensor,
    source_hw: tuple[int, int],
) -> torch.Tensor:
    """Map reliable target depths into the 105% source frustum from Appendix A.3."""
    h, w = depth.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((xx + 0.5, yy + 0.5, torch.ones_like(xx))).reshape(3, -1)
    xyz_target = torch.linalg.solve(target_K, pixels) * depth.reshape(1, -1)
    source_from_target = torch.linalg.inv(target_from_source)
    xyz = source_from_target[:3, :3] @ xyz_target + source_from_target[:3, 3:4]
    projected = source_K @ xyz
    x, y = projected[:2] / xyz[2:].clamp_min(1e-8)
    sh, sw = source_hw
    inside = ((2 * x / sw - 1).abs() <= 1.05) & ((2 * y / sh - 1).abs() <= 1.05) & (xyz[2] > 0)
    return (inside & (depth.flatten() > 0) & torch.isfinite(depth.flatten())).view(1, 1, h, w)
