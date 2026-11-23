"""Build calibrated source/target pairs from the official WildRGB-D NVS scene split."""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

PAIRS_PER_SCENE = 4


def write_json(file, record):
    """Replace a manifest atomically after writing it in full."""
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(file)


def sample_evenly(items, n):
    """Choose the same n elements independently of iteration order."""
    if len(items) < n:
        raise ValueError("Insufficient independent scenes or ten-frame groups")
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, n).round()]


def compare_projection(source, target, K, transform):
    """How much source depth is seen at the target, including out-of-frustum samples."""
    yy, xx = np.mgrid[3 : source.shape[0] : 6, 3 : source.shape[1] : 6]
    z = source[yy, xx]
    valid = np.isfinite(z) & (z > 0)
    ys, xs, z = yy[valid], xx[valid], z[valid]
    if not len(z):
        return 0.0, 0.0
    xy = np.stack(((xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z))
    xyz = transform[:3, :3] @ xy + transform[:3, 3:4]
    front = xyz[2] > 0
    uvw = K @ xyz[:, front]
    u, v = np.rint(uvw[:2] / uvw[2]).astype(int)
    inside = (u >= 0) & (v >= 0) & (u < target.shape[1]) & (v < target.shape[0])
    measured = target[v[inside], u[inside]]
    estimated = uvw[2, inside]
    jointly_valid = measured > 0
    ok = jointly_valid & (abs(estimated - measured) <= np.maximum(0.05, 0.05 * measured))
    return float(ok.sum() / len(z)), float(
        np.median(abs(estimated[ok] - measured[ok]))
    ) if ok.any() else 0.0


def build_scene(zf, scene_dir, split, output, *, allow_incomplete=False, scene_id=None):
    """Pick calibrated pairs from different ten-frame groups in one sequence."""
    scene_id = scene_id or Path(scene_dir).name
    meta = json.loads(zf.read(scene_dir + "/metadata"))
    h, w = meta["h"], meta["w"]
    K = np.array(meta["K"]).reshape(3, 3).T
    assert (
        np.allclose(K[-1], [0, 0, 1])
        and abs(K[0, 2] - w / 2) < 0.2 * w
        and abs(K[1, 2] - h / 2) < 0.2 * h
    )
    trajectory = np.loadtxt(io.BytesIO(zf.read(scene_dir + "/cam_poses.txt")))
    poses = {int(r[0]): r[1:].reshape(4, 4) for r in trajectory}
    frames = sorted(poses)
    groups = [frames[i : i + 10] for i in range(0, len(frames) - 9, 10)]
    selected = []
    chosen = set()

    def read_depth(frame):
        with zf.open(f"{scene_dir}/depth/{frame:05d}.png") as stream, Image.open(stream) as im:
            assert im.size == (w, h)
            return np.asarray(im).astype(np.float32) * 0.001

    def extract(frame, kind):
        member = f"{scene_dir}/{kind}/{frame:05d}.png"
        rel = f"frames/{scene_id}/{kind}/{frame:05d}.png"
        out = output / rel
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(zf, "link_to"):
                zf.link_to(member, out)  # Validated remote data already has an on-disk copy.
            else:
                payload = zf.read(member)  # zipfile checks the source CRC
                tmp = out.with_suffix(".partial")
                tmp.write_bytes(payload)
                tmp.replace(out)
        return rel

    count = min(len(groups), PAIRS_PER_SCENE) if allow_incomplete else PAIRS_PER_SCENE
    for preferred in sample_evenly(list(range(len(groups))), count):
        result = None
        for idx in sorted(range(len(groups)), key=lambda i: (abs(i - preferred), i)):
            if idx in chosen:
                continue
            group = groups[idx]
            frame = group[0]
            depth = read_depth(frame)
            for target in sorted(
                group[1:], key=lambda j: -np.linalg.norm(poses[j][:3, 3] - poses[frame][:3, 3])
            ):
                T = np.linalg.solve(poses[target], poses[frame])  # world_from_camera, OpenCV axes
                baseline = float(np.linalg.norm(T[:3, 3]))
                if (
                    baseline < 0.008
                    or not np.allclose(T[3], [0, 0, 0, 1])
                    or abs(np.linalg.det(T[:3, :3]) - 1) > 0.005
                ):
                    continue
                target_depth = read_depth(target)
                ov1, e1 = compare_projection(depth, target_depth, K, T)
                if ov1 <= 0.6:
                    continue
                ov2, e2 = compare_projection(target_depth, depth, K, np.linalg.inv(T))
                if ov2 <= 0.6:
                    continue
                chosen.add(idx)
                result = dict(
                    scene=scene_id,
                    split=split,
                    source=extract(frame, "rgb"),
                    target=extract(target, "rgb"),
                    source_depth=extract(frame, "depth"),
                    target_depth=extract(target, "depth"),
                    K_source=K.tolist(),
                    K_target=K.tolist(),
                    target_from_source=T.tolist(),
                    width=w,
                    height=h,
                    source_frame=frame,
                    target_frame=target,
                    baseline_m=baseline,
                    overlap_forward=ov1,
                    overlap_reverse=ov2,
                    median_consistent_depth_residual_m=[e1, e2],
                )
                break
            if result:
                break
        if result is None:
            if allow_incomplete:
                break  # Every unused group has been tested with the same geometry criteria.
            raise RuntimeError(f"Unable to find four distinct calibrated pairs: {scene_dir}")
        selected.append(result)
    return selected
