"""Generate an independent, small RGB-D multiview corpus for a two-stage SHARP pilot.

Run on CPU with Open3D's Embree ray caster. RGB and camera-axis depth come from the
same first intersection. Each scene has a persistent 3D layout, albedo, and lighting.
No SHARP, DA3 or photographs are used to generate the targets.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


def unit(v):
    """Normalize one nonzero camera or light direction."""
    return v / np.linalg.norm(v)


def make_texture(rng, kind):
    """Make tileable albedo with random grain, masonry, stone, fabric, or smooth paint."""
    side = 256
    y, x = np.mgrid[:side, :side].astype(np.float32) / side
    noise = np.zeros((side, side), np.float32)
    for freq, amp in ((4, 0.42), (8, 0.26), (20, 0.18), (48, 0.08)):
        grid = rng.uniform(-1, 1, (freq, freq)).astype(np.float32)
        noise += amp * cv2.remap(
            grid, x * freq, y * freq, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP
        )
    phase = rng.uniform(0, 2 * np.pi)
    if kind == "wood":
        base = np.array([0.56, 0.32, 0.14]) * rng.uniform(0.55, 1.42, 3)
        pattern = 0.90 + 0.14 * np.sin(2 * np.pi * (8 * y + 0.7 * noise) + phase) + 0.32 * noise
    elif kind == "brick":
        base = np.array([0.50, 0.30, 0.20]) * rng.uniform(0.6, 1.4, 3)
        u, v = (x * 4 + 0.5 * (np.floor(y * 8) % 2)) % 1, (y * 8) % 1
        mortar = (np.minimum(u, 1 - u) < 0.011) | (np.minimum(v, 1 - v) < 0.026)
        pattern = 1 + 0.45 * noise + 0.35 * mortar
    elif kind == "fabric":
        base = rng.uniform(0.18, 0.75, 3)
        bands = (np.floor(8 * x) + np.floor(8 * y)) % 2
        pattern = 0.72 + 0.3 * bands + 0.08 * np.sin(2 * np.pi * 32 * y) + 0.22 * noise
    else:
        base = (
            rng.uniform(0.15, 0.75, 3)
            if kind == "paint"
            else np.repeat(rng.uniform(0.35, 0.78), 3) * rng.uniform(0.85, 1.1, 3)
        )
        pattern = 1 + noise * (0.12 if kind == "paint" else 0.65)
    texture = (255 * np.clip(base * pattern[:, :, None], 0.01, 1)).astype(np.uint8)
    size = rng.uniform(0.25, 0.90) if kind != "brick" else rng.uniform(0.65, 1.5)
    return texture, size


class Room:
    """Render three calibrated views of a seeded room with persistent surface materials."""

    def __init__(self, seed):
        """Sample metric dimensions, furniture, geometry, albedo, and lighting."""
        rng = np.random.default_rng(seed)
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=3)
        self.materials = {}
        self.half_width = rng.uniform(2.1, 3.5)
        self.back = rng.uniform(4.0, 6.6)
        ceiling = rng.uniform(2.5, 3.4)
        self.light = unit(
            np.array([rng.uniform(-0.8, 0.8), -1, rng.uniform(-0.9, 0.25)], np.float32)
        )
        self.ambient = rng.uniform(0.44, 0.64)
        self.exposure = rng.uniform(0.82, 1.18)
        self.rng = rng

        self.box((0, 0.10, 2.2), (2 * self.half_width + 0.6, 0.2, 13), "wood")
        self.box((0, -ceiling - 0.10, 2.2), (2 * self.half_width + 0.6, 0.2, 13), "stone")
        self.box(
            (0, -ceiling / 2, self.back + 0.10),
            (2 * self.half_width + 0.6, ceiling + 0.4, 0.2),
            rng.choice(["brick", "paint", "stone"]),
        )
        for sign in (-1, 1):
            self.box(
                (sign * (self.half_width + 0.10), -ceiling / 2, 2.2),
                (0.2, ceiling + 0.4, 13),
                "paint",
            )
        for _ in range(rng.integers(2, 6)):
            self.box(
                (
                    rng.uniform(-self.half_width + 0.4, self.half_width - 0.4),
                    rng.uniform(-2.1, -0.6),
                    self.back - 0.05,
                ),
                (rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.7), 0.07),
                rng.choice(["fabric", "wood", "stone"]),
            )

        tx, tz = rng.uniform(-0.65, 0.65), rng.uniform(2.3, 3.5)
        tw, td, th = rng.uniform(1.2, 2.1), rng.uniform(0.65, 1.15), rng.uniform(0.63, 0.98)
        self.box((tx, -th, tz), (tw, 0.08, td), "wood")
        for sx in (-1, 1):
            for sz in (-1, 1):
                self.box(
                    (tx + sx * (tw / 2 - 0.07), -th / 2, tz + sz * (td / 2 - 0.07)),
                    (0.075, th, 0.075),
                    "wood",
                )
        for i in range(3):
            position = (
                tx + (i - 1) * tw * 0.31,
                -th - 0.04,
                tz + rng.uniform(-td * 0.30, td * 0.30),
            )
            self.object_at(position, rng.uniform(0.13, 0.34))

        for _ in range(rng.integers(5, 10)):
            x = rng.uniform(-self.half_width + 0.3, self.half_width - 0.3)
            z = rng.uniform(1.55, self.back - 0.35)
            if abs(x - tx) < tw * 0.55 and abs(z - tz) < td * 0.60:
                continue
            self.object_at((x, 0, z), rng.uniform(0.21, 0.62))

    def register(self, mesh, kind):
        """Add the mesh to the acceleration structure and assign its fixed material."""
        geometry_id = self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.materials[geometry_id] = make_texture(self.rng, kind)

    def box(self, center, dimensions, material, yaw=0.0):
        """Create an oriented cuboid using dimensions in metres."""
        mesh = o3d.geometry.TriangleMesh.create_box(*map(float, dimensions))
        mesh.translate(-np.array(dimensions) / 2)
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, yaw, 0)), center=(0, 0, 0))
        mesh.translate(center)
        self.register(mesh, material)

    def object_at(self, bottom, size):
        """Place a solid object in contact with a tabletop or the floor."""
        rng = self.rng
        kind = rng.choice(["box", "sphere", "cylinder"])
        material = rng.choice(["wood", "paint", "fabric", "stone"])
        h = size * rng.uniform(0.75, 1.8)
        position = np.array(bottom) - [0, h / 2, 0]
        if kind == "box":
            self.box(
                position,
                (size, h, size * rng.uniform(0.60, 1.30)),
                material,
                rng.uniform(-0.8, 0.8),
            )
            return
        if kind == "cylinder":
            mesh = o3d.geometry.TriangleMesh.create_cylinder(size / 2, h, resolution=36)
            mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)), center=(0, 0, 0))
        else:
            mesh = o3d.geometry.TriangleMesh.create_sphere(size / 2, resolution=20)
            verts = np.asarray(mesh.vertices)
            verts[:, 1] *= h / size
            verts[:, 2] *= rng.uniform(0.75, 1.15)
        mesh.translate(position)
        self.register(mesh, material)

    def cameras(self, w, h):
        """Sample intrinsics and three OpenCV camera-to-world transforms."""
        rng = self.rng
        fx = w * rng.uniform(0.80, 1.22)
        K = np.array(
            [
                [fx, 0, w / 2 + rng.uniform(-0.025, 0.025) * w],
                [0, fx, h / 2 + rng.uniform(-0.025, 0.025) * h],
                [0, 0, 1.0],
            ]
        )
        center = np.array(
            [rng.uniform(-0.5, 0.5), -rng.uniform(0.8, 1.55), rng.uniform(-0.14, 0.14)]
        )
        aim = np.array([rng.uniform(-0.25, 0.25), -rng.uniform(0.45, 1.05), rng.uniform(2.0, 3.2)])
        side = np.array(
            [
                rng.choice([-1, 1]) * rng.uniform(0.045, 0.23),
                rng.uniform(-0.055, 0.055),
                rng.uniform(-0.060, 0.060),
            ]
        )
        poses = []
        for step in (-1, 0, 1):
            eye = center + step * side
            look = unit(aim + step * side * rng.uniform(0.05, 0.9) - eye)
            right = unit(np.cross([0.0, 1.0, 0.0], look))
            down = np.cross(look, right)
            roll = rng.uniform(-0.025, 0.025)
            R = np.stack(
                [
                    right * np.cos(roll) + down * np.sin(roll),
                    down * np.cos(roll) - right * np.sin(roll),
                    look,
                ],
                axis=1,
            )
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = eye
            poses.append(T)
        return K, poses

    def render(self, K, pose, w, h):
        """Render sRGB and first-intersection camera-axis depth in millimetres."""
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        local = np.stack(
            ((xx + 0.5 - K[0, 2]) / K[0, 0], (yy + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(xx)),
            axis=2,
        )
        direction = (local @ pose[:3, :3].T).astype(np.float32)
        eye = np.broadcast_to(pose[:3, 3].astype(np.float32), (h, w, 3))
        rays = np.concatenate((eye, direction), axis=2)
        cast = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)
        z = cast["t_hit"].numpy()
        if not np.isfinite(z).all() or np.any(z <= 0) or np.max(z) > 65:
            raise ValueError("A training ray left the modeled room or exceeded depth image range.")
        ids = cast["geometry_ids"].numpy()
        normals = cast["primitive_normals"].numpy().copy()
        normals *= np.where((direction * normals).sum(axis=2, keepdims=True) > 0, -1.0, 1.0).astype(
            np.float32
        )
        pos = eye + direction * z[:, :, None]
        major = np.abs(normals).argmax(axis=2)
        u = np.where(major == 0, pos[:, :, 2], pos[:, :, 0]).astype(np.float32)
        v = np.where(major == 1, pos[:, :, 2], pos[:, :, 1]).astype(np.float32)
        albedo = np.empty((h, w, 3), np.float32)
        for gid in np.unique(ids):
            idx = ids == gid
            tex, meters = self.materials[gid]
            # Periodic coordinates anchored in world space retain texture in every camera.
            ui = (u[idx] / meters * tex.shape[1]).astype(int) % tex.shape[1]
            vi = (v[idx] / meters * tex.shape[0]).astype(int) % tex.shape[0]
            albedo[idx] = tex[vi, ui].astype(np.float32) / 255.0
        # Direction points toward an artificial ceiling light. Shadow rays stop just below
        # the ceiling so the room shell cannot block every ray.
        distance = np.maximum(0.08, (-2.30 - pos[:, :, 1]) / min(self.light[1], -0.02))
        light_dir = np.broadcast_to(self.light, (h, w, 3))
        shadow_rays = np.concatenate((pos + normals * 0.001, light_dir), axis=2).astype(np.float32)
        nearest = self.scene.cast_rays(o3d.core.Tensor(shadow_rays), nthreads=3)["t_hit"].numpy()
        shadow = nearest < distance
        diffuse = np.maximum(0.0, (normals * light_dir).sum(axis=2)) * (1 - 0.72 * shadow)
        lit = (
            (albedo**2.2)
            * self.exposure
            * (self.ambient + (1.35 - self.ambient) * diffuse[:, :, None])
        )
        rgb = np.rint(np.clip(lit, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        return rgb, np.rint(z * 1000).astype(np.uint16)


def generate(output, train_scenes, val_scenes, w, h):
    """Write separate scene seeds, undistorted views, depth, and pair metadata."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for split, count, base in (("train", train_scenes, 231104), ("val", val_scenes, 10000001)):
        for i in range(count):
            seed = base + i
            name = f"room_{seed}"
            folder = output / "frames" / name
            folder.mkdir(parents=True, exist_ok=True)
            room = Room(seed)
            K, poses = room.cameras(w, h)
            for j, pose in enumerate(poses):
                rgb, depth = room.render(K, pose, w, h)
                for kind, array in (("rgb", rgb), ("depth", depth)):
                    Image.fromarray(array).save(folder / f"{j}-{kind}.png", compress_level=1)
            for source, target in ((0, 1), (2, 1)):
                T = np.linalg.solve(poses[target], poses[source])
                src = f"frames/{name}/{source}"
                tgt = f"frames/{name}/{target}"
                records.append(
                    dict(
                        scene=name,
                        split=split,
                        source=src + "-rgb.png",
                        target=tgt + "-rgb.png",
                        source_depth=src + "-depth.png",
                        target_depth=tgt + "-depth.png",
                        K_source=K.tolist(),
                        K_target=K.tolist(),
                        target_from_source=T.tolist(),
                        width=w,
                        height=h,
                        source_frame=source,
                        target_frame=target,
                        baseline_m=float(np.linalg.norm(T[:3, 3])),
                    )
                )
            if i % 10 == 0 or i == count - 1:
                print(f"{split}: {i + 1}/{count} synthetic scenes", flush=True)
    manifest = dict(
        dataset="Procedural RGB-D pilot",
        width=w,
        height=h,
        depth_units="mm",
        convention="Metric OpenCV, T_target_from_source; pixel centers at x+0.5,y+0.5.",
        source="Open3D CPU first-hit ray casting with scene-fixed texture and lighting",
        pairs=records,
    )
    tmp = output / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output / "manifest.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--train-scenes", type=int, default=80)
    cli.add_argument("--val-scenes", type=int, default=16)
    cli.add_argument("--width", type=int, default=512)
    cli.add_argument("--height", type=int, default=384)
    args = cli.parse_args()
    if min(args.train_scenes, args.val_scenes, args.width, args.height) < 1:
        cli.error("Split sizes and image dimensions must be positive.")
    generate(args.output, args.train_scenes, args.val_scenes, args.width, args.height)
