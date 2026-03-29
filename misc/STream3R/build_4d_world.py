#!/usr/bin/env python3
"""Build a 4D world: static background point cloud + dynamic foreground objects + camera visualization.

Usage:
    python misc/STream3R/build_4d_world.py \
        --bg-videos bg1.mp4,bg2.mp4,bg3.mp4 \
        --fg-video fg.mp4 \
        --fg-prompt "person" \
        --output-dir outputs/4d_world \
        --bg-frames "0-80" \
        --fg-frames "0-260" \
        --sample-rate 8
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image as PILImage

# Ensure project root and STream3R dir are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
STREAM3R_DIR = os.path.dirname(os.path.abspath(__file__))
if STREAM3R_DIR not in sys.path:
    sys.path.insert(0, STREAM3R_DIR)

from infer_stream3r import extract_frames_from_video, parse_frame_indices, frames_to_preprocessed_tensor
from infer_stream3r_4d import make_pcd, save_glb, _align_sam3_masks
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri


def _parse_bg_fg_extract(spec_str: str) -> list[tuple[str, tuple[int, int]]]:
    """Parse bg-fg extraction spec like 'dog,0-64.human,0-32'.

    Returns list of (name, (start_frame, end_frame)).
    """
    results = []
    for item in spec_str.split("."):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid bg-fg-extract spec '{item}', expected 'name,start-end'")
        name = parts[0].strip()
        range_str = parts[1].strip()
        if "-" in range_str:
            start, end = range_str.split("-", 1)
            results.append((name, (int(start), int(end))))
        else:
            idx = int(range_str)
            results.append((name, (idx, idx)))
    return results


class DiskFrameStore:
    """Store per-frame numpy arrays on disk to avoid OOM. Access via indexing."""

    def __init__(self, tmpdir: str, prefix: str):
        self._dir = tmpdir
        self._prefix = prefix
        self._shapes = {}
        self._dtypes = {}

    def _path(self, idx: int) -> str:
        return os.path.join(self._dir, f"{self._prefix}_{idx:05d}.npy")

    def __setitem__(self, idx: int, arr: np.ndarray):
        np.save(self._path(idx), arr)
        self._shapes[idx] = arr.shape
        self._dtypes[idx] = arr.dtype

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.load(self._path(idx))

    def __contains__(self, idx: int) -> bool:
        return idx in self._shapes

    def __len__(self) -> int:
        return len(self._shapes)

    def shape(self, idx: int):
        return self._shapes[idx]


FRUSTUM_THUMB_SIZE = (256, 144)  # width, height for frustum image thumbnails


def _make_thumbnail(frame: np.ndarray, size=FRUSTUM_THUMB_SIZE) -> np.ndarray:
    """Resize frame to thumbnail for frustum texture (saves memory)."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def is_black_frame(frame: np.ndarray, threshold: float = 5.0) -> bool:
    """Check if a frame is (nearly) all black."""
    return frame.mean() < threshold


def feed_frames_to_session(
    session: StreamSession,
    images: torch.Tensor,
    label: str = "",
    stores: tuple[DiskFrameStore, DiskFrameStore, DiskFrameStore] | None = None,
    store_offset: int = 0,
) -> tuple[list[np.ndarray] | DiskFrameStore, list[np.ndarray] | DiskFrameStore, list[np.ndarray] | DiskFrameStore]:
    """Feed frames into an existing STream3R session and return per-frame results.

    The session accumulates state across calls, so all frames share one coordinate system.

    If `stores` is provided, saves results to disk instead of keeping in memory.

    Returns:
        frame_points, frame_colors, frame_confs
    """
    use_disk = stores is not None
    if not use_disk:
        frame_points = []
        frame_colors = []
        frame_confs = []
    else:
        pts_store, col_store, conf_store = stores

    n = images.shape[0]

    with torch.no_grad():
        for i in range(n):
            image = images[i : i + 1]
            predictions = session.forward_stream(image)

            preds = {}
            for key, val in predictions.items():
                if isinstance(val, torch.Tensor):
                    preds[key] = val.cpu().numpy().squeeze(0)
                else:
                    preds[key] = val

            wp = preds["world_points"]
            while wp.ndim > 3:
                wp = wp[-1]
            conf = preds.get("world_points_conf", np.ones(wp.shape[:-1]))
            while conf.ndim > 2:
                conf = conf[-1]
            img = preds["images"]
            while img.ndim > 3:
                img = img[-1]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            if use_disk:
                idx = store_offset + i
                pts_store[idx] = wp
                col_store[idx] = img
                conf_store[idx] = conf
            else:
                frame_points.append(wp)
                frame_colors.append(img)
                frame_confs.append(conf)

            if i % 10 == 0 or i == n - 1:
                print(f"  [{label}] Frame {i}/{n-1}: points {wp.shape}")

    if use_disk:
        return pts_store, col_store, conf_store
    return frame_points, frame_colors, frame_confs


def extract_poses_from_session(session: StreamSession, image_size_hw: tuple[int, int]) -> np.ndarray:
    """Extract all camera poses from the session's accumulated predictions.

    Returns:
        poses: (S, 4, 4) world-to-camera extrinsics for all frames fed so far
    """
    predictions = session.get_all_predictions()
    pose_enc = predictions["pose_enc"]  # (1, S, 9) tensor
    if isinstance(pose_enc, np.ndarray):
        pose_enc = torch.from_numpy(pose_enc)
    if pose_enc.ndim == 2:
        pose_enc = pose_enc.unsqueeze(0)

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc.float(), image_size_hw=image_size_hw
    )
    # extrinsics: (1, S, 3, 4) -> (S, 4, 4)
    ext_np = extrinsics.squeeze(0).cpu().numpy()
    poses = np.zeros((ext_np.shape[0], 4, 4))
    poses[:, :3, :] = ext_np
    poses[:, 3, 3] = 1.0
    return poses


def create_camera_frustum(pose_w2c, color, size=0.05, aspect=16/9, z_push=0.0, image_scale=1.0):
    """Create a camera frustum line set from a world-to-camera extrinsic matrix.

    z_push: shift frustum along the camera's viewing direction (Z axis in cam space).
            Positive = push towards the scene.
    image_scale: scale factor for the near plane (>1 = larger frustum face).
    """
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    hw = size * aspect / 2 * image_scale
    hh = size / 2 * image_scale
    d = size
    corners_cam = np.array([
        [0, 0, z_push],
        [-hw, -hh, d + z_push],
        [hw, -hh, d + z_push],
        [hw, hh, d + z_push],
        [-hw, hh, d + z_push],
    ])

    corners_world = (R.T @ (corners_cam.T - t[:, None])).T

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    colors = [color] * len(lines)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def create_frustum_image_quad(pose_w2c, image: np.ndarray, size=0.05, aspect=16/9, z_push=0.0, image_scale=1.0, brightness=1.0):
    """Create a textured quad at the frustum's near plane showing the camera image.

    image: (H, W, 3) uint8 or float32 numpy array.
    image_scale: scale factor for the quad relative to the frustum near plane (>1 = larger).
    Returns (mesh, material) — an Open3D TriangleMesh + MaterialRecord with albedo texture.
    """
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    hw = size * aspect / 2 * image_scale
    hh = size / 2 * image_scale
    d = size
    # Near plane corners in camera space (same as frustum corners 1-4)
    corners_cam = np.array([
        [-hw, -hh, d + z_push],  # bottom-left
        [ hw, -hh, d + z_push],  # bottom-right
        [ hw,  hh, d + z_push],  # top-right
        [-hw,  hh, d + z_push],  # top-left
    ])
    corners_world = (R.T @ (corners_cam.T - t[:, None])).T

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners_world)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    # UV coords: map corners to image corners
    mesh.triangle_uvs = o3d.utility.Vector2dVector([
        [0, 1], [1, 1], [1, 0],  # triangle 0: BL, BR, TR
        [0, 1], [1, 0], [0, 0],  # triangle 1: BL, TR, TL
    ])

    # Convert image to Open3D Image for texture
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if brightness != 1.0:
        image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    mesh.textures = [o3d.geometry.Image(image)]

    # Per-triangle material index (all use texture 0)
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])

    # Create material with albedo texture
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.albedo_img = o3d.geometry.Image(image)
    mat.base_color = [1.0, 1.0, 1.0, 1.0]

    return mesh, mat


def smooth_poses(poses_w2c, window_size=5):
    """Smooth camera poses with a moving average on camera centers and forward directions.

    Args:
        poses_w2c: (N, 4, 4) w2c extrinsics
        window_size: smoothing window (must be odd)
    Returns:
        smoothed (N, 4, 4) w2c extrinsics
    """
    if len(poses_w2c) < 3 or window_size < 3:
        return poses_w2c

    n = len(poses_w2c)
    half = window_size // 2

    # Extract camera centers and forward/up directions
    centers = np.array([-p[:3, :3].T @ p[:3, 3] for p in poses_w2c])
    forwards = np.array([p[:3, :3].T @ np.array([0, 0, 1]) for p in poses_w2c])
    ups = np.array([p[:3, :3].T @ np.array([0, -1, 0]) for p in poses_w2c])

    # Smooth with uniform moving average
    def _smooth(arr):
        out = np.empty_like(arr)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = arr[lo:hi].mean(axis=0)
        return out

    smooth_centers = _smooth(centers)
    smooth_forwards = _smooth(forwards)
    smooth_ups = _smooth(ups)

    # Rebuild w2c from smoothed values
    result = np.zeros_like(poses_w2c)
    for i in range(n):
        fwd = smooth_forwards[i]
        fwd = fwd / np.linalg.norm(fwd)
        up = smooth_ups[i]
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)
        # R_w2c: rows are right, -up, forward in world coords
        R = np.stack([right, -up, fwd], axis=0)
        t = -R @ smooth_centers[i]
        result[i, :3, :3] = R
        result[i, :3, 3] = t
        result[i, 3, 3] = 1.0
    return result


def create_camera_path(poses_w2c, color, z_push=0.0):
    """Create a line set showing camera trajectory."""
    centers = []
    for pose in poses_w2c:
        R = pose[:3, :3]
        t = pose[:3, 3]
        C = -R.T @ t
        if z_push != 0.0:
            # Push along camera's Z (viewing) direction in world space
            forward_world = R.T @ np.array([0, 0, z_push])
            C = C + forward_world
        centers.append(C)

    if len(centers) < 2:
        return o3d.geometry.LineSet()

    lines = [[i, i + 1] for i in range(len(centers) - 1)]
    colors = [color] * len(lines)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(centers))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def render_world_video(
    output_dir: str,
    per_frame_bg_points: list[np.ndarray],
    per_frame_bg_colors: list[np.ndarray],
    per_frame_fg_points: list[np.ndarray],
    per_frame_fg_colors: list[np.ndarray],
    camera_pose_tracks: list[dict],
    frustum_size: float = 0.05,
    frustum_z_push: float = 0.0,
    fps: float = 8.0,
    render_size: tuple[int, int] = (1920, 1080),
    cam_az: float = 0.0,
    cam_el: float = 20.0,
    cam_dist_scale: float = 1.0,
    point_size: float = 2.0,
    fov: float = 60.0,
    frustum_image_scale: float = 2.0,
    frustum_image_brightness: float = 1.0,
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> str:
    """Render the 4D world: static bg + dynamic fg + animated camera viz -> video.

    camera_pose_tracks: list of dicts with keys:
        - "poses": (N, 4, 4) w2c extrinsics
        - "color": [r, g, b]
        - "num_frames": number of frames for this track
        - "global_start": frame offset relative to fg frame timeline (for bg, typically 0)
    """
    num_frames = len(per_frame_fg_points)
    W, H = render_size

    # Save debug log to file
    import builtins
    _log_path = os.path.join(output_dir, "render_debug.log")
    _log_file = open(_log_path, "w")
    _builtin_print = builtins.print
    def print(*args, **kwargs):
        _builtin_print(*args, **kwargs)
        kwargs.pop("file", None)
        _builtin_print(*args, file=_log_file, **kwargs)
        _log_file.flush()

    # ---- Collect ALL frustum corner points (the hard boundary we must fit) ----
    aspect = W / H
    hw = frustum_size * aspect / 2 * frustum_image_scale
    hh = frustum_size / 2 * frustum_image_scale
    d_f = frustum_size
    zp = frustum_z_push
    corners_cam = np.array([
        [0, 0, zp],
        [-hw, -hh, d_f + zp],
        [hw, -hh, d_f + zp],
        [hw, hh, d_f + zp],
        [-hw, hh, d_f + zp],
    ])
    identity = np.eye(4)
    frustum_pts = []  # frustum corners only — these MUST be visible
    for track in camera_pose_tracks:
        poses = track["poses"]  # already filtered to valid poses
        for p in poses:
            if np.allclose(p, identity, atol=1e-6):
                continue
            R = p[:3, :3]
            t = p[:3, 3]
            c2w_pos = (-R.T @ t).reshape(1, 3)
            frustum_pts.append(c2w_pos)
            cw = (R.T @ (corners_cam.T - t[:, None])).T
            frustum_pts.append(cw)
    n_frustum_pts = sum(len(p) for p in frustum_pts)
    n_valid_poses = sum(1 for p_arr in frustum_pts if len(p_arr) == 1)  # camera centers = 1pt each
    print(f"  Frustum points collected: {n_frustum_pts} pts from {len(camera_pose_tracks)} tracks ({n_valid_poses} valid poses)")
    for ti, track in enumerate(camera_pose_tracks):
        poses = track["poses"]
        n_id = sum(1 for p in poses if np.allclose(p, identity, atol=1e-6))
        print(f"    Track {ti}: {len(poses)} poses, {n_id} identity (skipped), color={track['color']}, no_trail={track.get('no_trail', False)}")

    # Point cloud points (use percentile to remove point cloud outliers only)
    all_bg = np.concatenate([p for p in per_frame_bg_points if len(p) > 0], axis=0) if any(len(p) > 0 for p in per_frame_bg_points) else np.empty((0, 3))
    all_cloud = [all_bg] + [p for p in per_frame_fg_points if len(p) > 0]
    cloud_pts = np.concatenate(all_cloud, axis=0) if all_cloud else np.empty((0, 3))
    del all_bg
    if len(cloud_pts) > 0:
        lo = np.percentile(cloud_pts, 2, axis=0)
        hi = np.percentile(cloud_pts, 98, axis=0)
        mask = np.all((cloud_pts >= lo) & (cloud_pts <= hi), axis=1)
        cloud_pts = cloud_pts[mask] if mask.sum() > 100 else cloud_pts

    # Combine: frustum corners (no filtering!) + filtered point cloud
    frustum_all = np.concatenate(frustum_pts, axis=0) if frustum_pts else np.empty((0, 3))
    scene_pts = np.concatenate([cloud_pts, frustum_all], axis=0) if len(cloud_pts) > 0 else frustum_all
    scene_center = np.mean(scene_pts, axis=0)

    print(f"  [DEBUG] cloud_pts: {len(cloud_pts)}, frustum_all: {len(frustum_all)}, scene_pts: {len(scene_pts)}")
    print(f"  [DEBUG] cloud  range: {cloud_pts.min(axis=0)} ~ {cloud_pts.max(axis=0)}" if len(cloud_pts) > 0 else "  [DEBUG] cloud: empty")
    print(f"  [DEBUG] frustum range: {frustum_all.min(axis=0)} ~ {frustum_all.max(axis=0)}" if len(frustum_all) > 0 else "  [DEBUG] frustum: empty")
    print(f"  [DEBUG] scene_center: {scene_center}")
    print(f"  [DEBUG] frustum_size={frustum_size}, image_scale={frustum_image_scale}, z_push={frustum_z_push}")
    print(f"  [DEBUG] frustum hw={hw:.6f}, hh={hh:.6f}, d_f={d_f:.6f}")

    # Compute render camera direction from frustum poses:
    # Average the frustum camera "forward" directions and positions,
    # then place the render camera behind them (looking toward scene center) with elevation.
    identity = np.eye(4)
    frustum_positions = []
    frustum_forwards = []
    for track in camera_pose_tracks:
        poses = track["poses"]
        for p in poses:
            if np.allclose(p, identity, atol=1e-6):
                continue
            R = p[:3, :3]
            t = p[:3, 3]
            cam_center = -R.T @ t
            # Forward direction in world coords: the camera looks along +Z in cam space
            cam_fwd = R.T @ np.array([0, 0, 1.0])
            frustum_positions.append(cam_center)
            frustum_forwards.append(cam_fwd)

    if len(frustum_positions) > 0:
        avg_cam_pos = np.mean(frustum_positions, axis=0)
        avg_cam_fwd = np.mean(frustum_forwards, axis=0)
        avg_cam_fwd = avg_cam_fwd / (np.linalg.norm(avg_cam_fwd) + 1e-8)
        # Render camera looks FROM behind the frustum cameras TOWARD the scene center
        # base_dir = from avg_cam_pos toward scene_center
        base_dir = scene_center - avg_cam_pos
        if np.linalg.norm(base_dir) < 1e-6:
            base_dir = avg_cam_fwd
        base_dir = base_dir / np.linalg.norm(base_dir)
        # Apply az rotation around Y axis relative to this base direction
        az = np.radians(cam_az)
        # Rotation matrix around Y
        cos_a, sin_a = np.cos(az), np.sin(az)
        Ry = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        cam_dir_horiz = Ry @ base_dir
        # Apply elevation: rotate cam_dir_horiz to tilt upward (look down)
        el = np.radians(-cam_el)
        # Right vector (horizontal)
        w_up = np.array([0.0, -1.0, 0.0])
        right = np.cross(cam_dir_horiz, w_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / np.linalg.norm(right)
        # Rodrigues rotation around 'right' by elevation angle
        cos_e, sin_e = np.cos(el), np.sin(el)
        K = np.array([[0, -right[2], right[1]], [right[2], 0, -right[0]], [-right[1], right[0], 0]])
        R_el = np.eye(3) + sin_e * K + (1 - cos_e) * (K @ K)
        cam_dir = R_el @ cam_dir_horiz
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        print(f"  [AUTO-CAM] avg_frustum_pos={avg_cam_pos}, avg_fwd={avg_cam_fwd}")
        print(f"  [AUTO-CAM] base_dir={base_dir}, cam_dir={cam_dir} (az={cam_az}, el={cam_el})")
    else:
        # Fallback: use az/el directly
        az = np.radians(cam_az)
        el = np.radians(cam_el)
        cam_dir = np.array([
            np.sin(az) * np.cos(el),
            -np.sin(el),
            -np.cos(az) * np.cos(el),
        ])
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        print(f"  [AUTO-CAM] No frustum poses, using fallback az={cam_az}, el={cam_el}")

    # Helper: build full camera matrix for a given distance, project all scene_pts to 2D
    def _build_camera(dist):
        """Build extrinsic + intrinsic for render camera at given distance from scene_center."""
        pos = scene_center - dist * cam_dir
        fwd = cam_dir.copy()  # looking towards scene_center
        w_up = np.array([0.0, -1.0, 0.0])
        r = np.cross(fwd, w_up)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1.0, 0.0, 0.0])
        else:
            r = r / np.linalg.norm(r)
        u = np.cross(r, fwd)
        u = u / np.linalg.norm(u)
        Rot = np.stack([r, -u, fwd], axis=0)
        tvec = -Rot @ pos
        _fy = H / (2 * np.tan(np.radians(fov / 2)))
        _fx = _fy
        _cx, _cy = W / 2.0, H / 2.0
        return pos, Rot, tvec, _fx, _fy, _cx, _cy

    def _project(dist, pts):
        """Project 3D points to 2D pixel coords at given camera distance. Returns (px, py)."""
        pos, Rot, tvec, _fx, _fy, _cx, _cy = _build_camera(dist)
        # Transform to camera coords: p_cam = R @ p_world + t
        p_cam = (Rot @ pts.T).T + tvec  # (N, 3)
        # p_cam[:, 2] is depth (along forward direction)
        depth = p_cam[:, 2]
        valid = depth > 0.001
        if valid.sum() == 0:
            return None, None, valid
        px = _fx * p_cam[valid, 0] / depth[valid] + _cx
        py = _fy * p_cam[valid, 1] / depth[valid] + _cy
        return px, py, valid

    def _all_in_frame(dist, pts, margin=0):
        """Check if ALL points project within [margin, W-margin] x [margin, H-margin]."""
        px, py, valid = _project(dist, pts)
        if px is None:
            return False
        if valid.sum() < len(pts):
            return False  # some points behind camera
        return (px.min() >= margin and px.max() <= W - margin and
                py.min() >= margin and py.max() <= H - margin)

    # ---- Binary search for minimum camera distance ----
    # Start with a rough estimate, then binary search
    lo_dist = 0.01
    hi_dist = 0.1
    # Expand hi_dist until everything fits
    for _ in range(50):
        if _all_in_frame(hi_dist, scene_pts):
            break
        hi_dist *= 2.0
    else:
        print(f"  WARNING: could not find camera distance to fit scene (tried up to {hi_dist:.2f})")

    # Binary search between lo_dist and hi_dist
    for _ in range(50):
        mid = (lo_dist + hi_dist) / 2
        if _all_in_frame(mid, scene_pts):
            hi_dist = mid
        else:
            lo_dist = mid
        if hi_dist - lo_dist < 0.0001:
            break
    auto_dist = hi_dist

    # Debug: show actual 2D coverage for ALL points, and separately for frustum-only
    final_dist = auto_dist * cam_dist_scale
    px, py, valid = _project(final_dist, scene_pts)
    if px is not None:
        print(f"  [DEBUG] 2D ALL pts: x=[{px.min():.0f}, {px.max():.0f}] y=[{py.min():.0f}, {py.max():.0f}] "
              f"in {W}x{H}, coverage={((px.max()-px.min())/W*100):.1f}%x{((py.max()-py.min())/H*100):.1f}%")
    else:
        print(f"  [DEBUG] 2D ALL pts: FAILED to project (all behind camera?)")
    if len(frustum_all) > 0:
        fpx, fpy, fvalid = _project(final_dist, frustum_all)
        if fpx is not None:
            print(f"  [DEBUG] 2D FRUSTUM: x=[{fpx.min():.0f}, {fpx.max():.0f}] y=[{fpy.min():.0f}, {fpy.max():.0f}] "
                  f"coverage={((fpx.max()-fpx.min())/W*100):.1f}%x{((fpy.max()-fpy.min())/H*100):.1f}%")
        else:
            print(f"  [DEBUG] 2D FRUSTUM: FAILED to project")
    if len(cloud_pts) > 0:
        cpx, cpy, cvalid = _project(final_dist, cloud_pts)
        if cpx is not None:
            print(f"  [DEBUG] 2D CLOUD:   x=[{cpx.min():.0f}, {cpx.max():.0f}] y=[{cpy.min():.0f}, {cpy.max():.0f}] "
                  f"coverage={((cpx.max()-cpx.min())/W*100):.1f}%x{((cpy.max()-cpy.min())/H*100):.1f}%")
    print(f"  [DEBUG] cam_dir={cam_dir}, scene_center={scene_center}")
    print(f"  Auto camera: auto_dist={auto_dist:.4f}, cam_dist_scale={cam_dist_scale:.2f}, final_dist={final_dist:.4f}")

    cam_dist = auto_dist * cam_dist_scale
    cam_pos, Rot_final, t_final, fx, fy, cx, cy = _build_camera(cam_dist)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rot_final
    extrinsic[:3, 3] = t_final
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # Extract right/up/forward for later use
    right = Rot_final[0]
    up = -Rot_final[1]
    forward = Rot_final[2]

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(np.array([*bg_color, 1.0]))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 5.0

    fg_line_mat = o3d.visualization.rendering.MaterialRecord()
    fg_line_mat.shader = "unlitLine"
    fg_line_mat.line_width = 8.0

    n_bg = len(per_frame_bg_points)
    total_bg = sum(len(p) for p in per_frame_bg_points if len(p) > 0)
    print(f"  Background: {total_bg:,} pts across {n_bg} frames (progressive reveal)")
    bg_added_up_to = -1  # track which bg frames have been added
    bg_pts_accum = []
    bg_col_accum = []
    bg_dirty = False  # whether bg pcd needs rebuild

    video_out = os.path.join(output_dir, "4d_world.mp4")
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            video_out,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    print(f"\n=== Rendering {num_frames} frames ===")
    print(f"  Resolution: {W}x{H}, FPS: {fps}")
    print(f"  Camera: az={np.degrees(az):.0f}, el={np.degrees(el):.0f}, dist={cam_dist:.2f}, fov={fov:.0f}")
    print(f"  Camera pose tracks: {len(camera_pose_tracks)}")

    for i in range(num_frames):
        # Update background (progressive reveal)
        bg_frame = min(int(i * n_bg / num_frames), n_bg - 1)
        while bg_added_up_to < bg_frame:
            bg_added_up_to += 1
            if len(per_frame_bg_points[bg_added_up_to]) > 0:
                bg_pts_accum.append(per_frame_bg_points[bg_added_up_to])
                bg_col_accum.append(per_frame_bg_colors[bg_added_up_to])
                bg_dirty = True
        if bg_dirty:
            if renderer.scene.has_geometry("background"):
                renderer.scene.remove_geometry("background")
            if bg_pts_accum:
                bg_pcd = make_pcd(
                    np.concatenate(bg_pts_accum, axis=0),
                    np.concatenate(bg_col_accum, axis=0),
                )
                renderer.scene.add_geometry("background", bg_pcd, mat)
            bg_dirty = False

        # Update foreground
        if renderer.scene.has_geometry("foreground"):
            renderer.scene.remove_geometry("foreground")
        if len(per_frame_fg_points[i]) > 0:
            fg_pcd = make_pcd(per_frame_fg_points[i], per_frame_fg_colors[i])
            renderer.scene.add_geometry("foreground", fg_pcd, mat)

        # Update camera tracks
        for ti, track in enumerate(camera_pose_tracks):
            frustum_name = f"cam_frustum_{ti}"
            trail_name = f"cam_trail_{ti}"
            quad_name = f"cam_quad_{ti}"
            for gname in (frustum_name, trail_name, quad_name):
                if renderer.scene.has_geometry(gname):
                    renderer.scene.remove_geometry(gname)

            poses_t = track["poses"]
            color = track["color"]
            n_track = track["num_frames"]  # number of valid poses
            track_valid_mask = track.get("valid_mask", None)
            track_frames = track.get("frames", None)

            if track_valid_mask is not None:
                # FG track: fixed pose (first valid frame), visible only when non-black
                n_total = track.get("num_total", n_track)
                total_frame = min(int(i * n_total / num_frames), n_total - 1)
                if not track_valid_mask[total_frame]:
                    continue  # black frame — hide frustum & quad
                # Always show at first valid frame's pose; trail only if not suppressed
                use_mat = fg_line_mat if track.get("no_trail", False) else line_mat
                cur_pose = poses_t[0]
                frustum = create_camera_frustum(cur_pose, color, size=frustum_size, z_push=frustum_z_push, image_scale=frustum_image_scale)
                renderer.scene.add_geometry(frustum_name, frustum, use_mat)
                if not track.get("no_trail", False) and n_track >= 2:
                    trail = create_camera_path(poses_t, color, z_push=frustum_z_push)
                    renderer.scene.add_geometry(trail_name, trail, line_mat)
                # Image quad: show current valid frame's image
                if track_frames is not None:
                    # Map total_frame to valid frame index
                    valid_idx = sum(1 for k in range(total_frame) if track_valid_mask[k])
                    valid_idx = min(valid_idx, len(track_frames) - 1)
                    quad, quad_mat = create_frustum_image_quad(cur_pose, track_frames[valid_idx],
                                                               size=frustum_size, z_push=frustum_z_push,
                                                               image_scale=frustum_image_scale,
                                                               brightness=frustum_image_brightness)
                    renderer.scene.add_geometry(quad_name, quad, quad_mat)
            else:
                # BG track: animated frustum + growing trail
                track_frame = min(int(i * n_track / num_frames), n_track - 1)
                cur_pose = poses_t[track_frame]
                frustum = create_camera_frustum(cur_pose, color, size=frustum_size, z_push=frustum_z_push, image_scale=frustum_image_scale)
                renderer.scene.add_geometry(frustum_name, frustum, line_mat)
                if track_frame >= 1:
                    trail = create_camera_path(poses_t[:track_frame + 1], color, z_push=frustum_z_push)
                    renderer.scene.add_geometry(trail_name, trail, line_mat)
                # Image quad
                if track_frames is not None and track_frame < len(track_frames):
                    quad, quad_mat = create_frustum_image_quad(cur_pose, track_frames[track_frame],
                                                               size=frustum_size, z_push=frustum_z_push,
                                                               image_scale=frustum_image_scale,
                                                               brightness=frustum_image_brightness)
                    renderer.scene.add_geometry(quad_name, quad, quad_mat)

        renderer.setup_camera(intrinsic, extrinsic)
        img = np.asarray(renderer.render_to_image())
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # RGBA -> RGB
        if ffmpeg_proc.poll() is not None:
            stderr_out = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
            raise RuntimeError(f"ffmpeg exited early (code {ffmpeg_proc.returncode}): {stderr_out}")
        ffmpeg_proc.stdin.write(img.tobytes())

        if i % 10 == 0 or i == num_frames - 1:
            print(f"  Rendered frame {i}/{num_frames-1}")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        stderr_out = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
        print(f"  WARNING: ffmpeg exited with code {ffmpeg_proc.returncode}: {stderr_out}")
    print(f"Saved video: {video_out}")
    print(f"Debug log saved: {_log_path}")
    _log_file.close()
    return video_out


def main():
    parser = argparse.ArgumentParser(description="Build 4D world: static bg + dynamic fg + camera viz")

    # Input
    parser.add_argument("--bg-videos", required=True, help="Background videos (comma-separated)")
    parser.add_argument("--fg-video", required=True, help="Foreground video(s) (comma-separated)")
    parser.add_argument("--fg-prompt", required=True, help="Foreground prompt(s) (comma-separated, matches fg-video)")
    parser.add_argument("--bg-fg-extract", type=str, default=None,
                        help="Extract fg objects from bg video. Format: 'name,start-end.name2,start-end' "
                             "e.g. 'dog,0-64.human,0-32'. Frame numbers are sampled-frame indices.")
    parser.add_argument("--output-dir", "-o", default="outputs/4d_world")

    # Frame sampling
    parser.add_argument("--bg-frames", type=str, default=None, help="Frame range for bg videos (e.g. '0-80')")
    parser.add_argument("--fg-frames", type=str, default=None, help="Frame range for fg video (e.g. '0-260')")
    parser.add_argument("--sample-rate", type=int, default=8)

    # STream3R
    parser.add_argument("--model-path", type=str, default="hf_cache/yslan--STream3R")
    parser.add_argument("--stream-mode", type=str, default="window", choices=["causal", "window"])
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    # Point cloud
    parser.add_argument("--conf-percentile", type=float, default=30.0, help="Bg confidence threshold percentile")
    parser.add_argument("--fg-conf-percentile", type=float, default=0.0, help="Fg confidence threshold percentile")
    parser.add_argument("--mask-dilate", type=int, default=5)
    parser.add_argument("--bg-voxel-size", type=float, default=0.001)

    # Camera viz
    parser.add_argument("--frustum-size", type=float, default=0.05, help="Camera frustum size")
    parser.add_argument("--frustum-image-scale", type=float, default=2.0, help="Scale factor for frustum image quad relative to frustum size")
    parser.add_argument("--frustum-image-brightness", type=float, default=1.0, help="Brightness multiplier for frustum image (>1 brighter, <1 darker)")
    parser.add_argument("--frustum-z-push", type=float, default=0.0, help="Push frustums along viewing direction towards scene")
    parser.add_argument("--cam-smooth", type=int, default=0, help="Smoothing window size for BG camera poses (0=off, odd number e.g. 5,7,11)")
    parser.add_argument("--no-camera-viz", action="store_true", help="Skip camera visualization")
    parser.add_argument("--bg-cam-color", type=str, default=None, help="BG frustum RGB color e.g. '59,125,35'")
    parser.add_argument("--fg-cam-colors", type=str, default=None, help="FG frustum RGB colors separated by ';' e.g. '192,79,21;33,95,154'")
    parser.add_argument("--combine", action="store_true", help="Auto-combine 4D render with source bg/fg videos after rendering")
    parser.add_argument("--combine-border", type=int, default=10, help="Border thickness for combined video (pixels)")
    parser.add_argument("--combine-height", type=int, default=0, help="Target total height for combined video (0=auto from render_height)")

    # Render
    parser.add_argument("--render-width", type=int, default=1920)
    parser.add_argument("--render-height", type=int, default=1080)
    parser.add_argument("--render-fps", type=float, default=8.0)
    parser.add_argument("--cam-az", type=float, default=0.0)
    parser.add_argument("--cam-el", type=float, default=20.0)
    parser.add_argument("--cam-dist", type=float, default=1.0)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--fov", type=float, default=60.0)

    # Output
    parser.add_argument("--sphere-radius", type=float, default=0.001)
    parser.add_argument("--save-glb", action="store_true", help="Save GLB files")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bg_video_list = [v.strip() for v in args.bg_videos.split(",")]
    fg_video_list = [v.strip() for v in args.fg_video.split(",")]
    prompt_list = [p.strip() for p in args.fg_prompt.split(",")]

    # Pad to match lengths
    if len(fg_video_list) == 1 and len(prompt_list) > 1:
        fg_video_list = fg_video_list * len(prompt_list)
    if len(prompt_list) == 1 and len(fg_video_list) > 1:
        prompt_list = prompt_list * len(fg_video_list)
    assert len(fg_video_list) == len(prompt_list), \
        f"Number of fg videos ({len(fg_video_list)}) must match prompts ({len(prompt_list)})"

    bg_frame_indices = parse_frame_indices(args.bg_frames) if args.bg_frames else None
    fg_frame_indices = parse_frame_indices(args.fg_frames) if args.fg_frames else None

    bg_removal_prompts = [p.strip() for p in args.fg_prompt.split(",") if p.strip()]

    # ======== Step 0: Extract all frames from all videos ========
    print("=" * 60)
    print("Step 0: Extracting frames from all videos")
    print("=" * 60)

    # Structure: list of dicts per video
    # Each entry keeps ALL frames (including black) but marks which are valid
    video_entries = []

    # For bg videos: extract frames using fg_frame_indices (for pose alignment with fg),
    # but mark which frames fall within bg_frame_indices for point cloud extraction.
    for vi, bg_vid in enumerate(bg_video_list):
        # Use fg frame range so bg camera trajectory matches fg timeline
        use_indices = fg_frame_indices if fg_frame_indices else bg_frame_indices
        all_frames = extract_frames_from_video(bg_vid, sample_rate=args.sample_rate, frame_indices=use_indices)
        valid_mask = [not is_black_frame(f) for f in all_frames]
        valid_frames = [f for f, v in zip(all_frames, valid_mask) if v]
        n_skipped = len(all_frames) - len(valid_frames)

        # Determine which frames are within bg_frame_indices (for point cloud only)
        if use_indices and bg_frame_indices:
            sampled_indices = sorted(use_indices)[::args.sample_rate]
            bg_point_mask = [idx in bg_frame_indices for idx in sampled_indices[:len(all_frames)]]
        else:
            bg_point_mask = [True] * len(all_frames)

        # Compute sampled original frame indices for bg-fg extraction mapping
        if use_indices:
            sampled_orig_indices = sorted(use_indices)[::args.sample_rate][:len(all_frames)]
        else:
            sampled_orig_indices = list(range(0, len(all_frames) * args.sample_rate, args.sample_rate))

        print(f"  BG {vi}: {bg_vid} -> {len(all_frames)} extracted, {n_skipped} black skipped, {len(valid_frames)} valid, {sum(bg_point_mask)} in bg-frames range")
        video_entries.append({
            "path": bg_vid, "all_frames": all_frames, "valid_mask": valid_mask,
            "valid_frames": valid_frames, "label": f"bg_{vi}",
            "is_bg": True, "video_idx": vi,
            "num_total": len(all_frames),
            "bg_point_mask": bg_point_mask,
            "sampled_orig_indices": sampled_orig_indices,
        })

    for fi, fg_vid in enumerate(fg_video_list):
        all_frames = extract_frames_from_video(fg_vid, sample_rate=args.sample_rate, frame_indices=fg_frame_indices)
        valid_mask = [not is_black_frame(f) for f in all_frames]
        valid_frames = [f for f, v in zip(all_frames, valid_mask) if v]
        n_skipped = len(all_frames) - len(valid_frames)
        print(f"  FG {fi}: {fg_vid} -> {len(all_frames)} extracted, {n_skipped} black skipped, {len(valid_frames)} valid")
        video_entries.append({
            "path": fg_vid, "all_frames": all_frames, "valid_mask": valid_mask,
            "valid_frames": valid_frames, "label": f"fg_{fi}",
            "is_bg": False, "video_idx": fi, "prompt": prompt_list[fi],
            "num_total": len(all_frames),
        })

    total_valid = sum(len(e["valid_frames"]) for e in video_entries)
    print(f"\nTotal valid frames to process: {total_valid}")

    # ======== Step 1: Run all frames through ONE STream3R session ========
    print("\n" + "=" * 60)
    print("Step 1: STream3R inference (single shared session)")
    print("=" * 60)

    # Create temp dir for disk-backed frame stores
    _tmpdir = tempfile.mkdtemp(prefix="build4d_")
    print(f"  Temp dir for frame data: {_tmpdir}")
    pts_store = DiskFrameStore(_tmpdir, "pts")
    col_store = DiskFrameStore(_tmpdir, "col")
    conf_store = DiskFrameStore(_tmpdir, "conf")

    model = STream3R.from_pretrained(args.model_path).to(args.device)
    model.eval()
    session = StreamSession(model, mode=args.stream_mode, window_size=args.window_size)
    print(f"  Mode: {args.stream_mode}" + (f", window_size: {args.window_size}" if args.stream_mode == "window" else ""))

    # Feed only valid (non-black) frames through STream3R, save to disk
    global_offset = 0
    H_s, W_s = None, None
    for entry in video_entries:
        valid_frames = entry["valid_frames"]
        if len(valid_frames) == 0:
            entry["global_start"] = global_offset
            entry["num_valid"] = 0
            del entry["valid_frames"]
            continue
        images = frames_to_preprocessed_tensor(valid_frames, device=args.device)
        feed_frames_to_session(
            session, images, label=entry["label"],
            stores=(pts_store, col_store, conf_store), store_offset=global_offset,
        )
        entry["global_start"] = global_offset
        entry["num_valid"] = len(valid_frames)
        if H_s is None:
            H_s, W_s = pts_store.shape(global_offset)[:2]
        global_offset += len(valid_frames)
        del images
        del entry["valid_frames"]
        torch.cuda.empty_cache()

    # Extract all poses at once (all valid frames in same coordinate system)
    all_poses = extract_poses_from_session(session, image_size_hw=(H_s, W_s))
    print(f"\nTotal poses: {all_poses.shape[0]} (should be {global_offset})")

    # Build per-entry pose arrays (expand valid-only to full timeline)
    for entry in video_entries:
        valid_mask = entry["valid_mask"]
        num_total = entry["num_total"]
        start = entry["global_start"]
        valid_poses = all_poses[start:start + entry["num_valid"]]
        full_poses = np.tile(np.eye(4), (num_total, 1, 1))
        vi_idx = 0
        for fi in range(num_total):
            if valid_mask[fi]:
                full_poses[fi] = valid_poses[vi_idx]
                vi_idx += 1
        entry["poses"] = full_poses

        # Build mapping: full frame index -> disk store index (or -1 for black frames)
        store_map = []
        vi_idx = 0
        for fi in range(num_total):
            if valid_mask[fi]:
                store_map.append(start + vi_idx)
                vi_idx += 1
            else:
                store_map.append(-1)
        entry["store_map"] = store_map
        entry["num_frames"] = num_total

    del model, session
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Disk frame store: {len(pts_store)} frames saved to {_tmpdir}")

    # ======== Step 2: SAM3 segmentation + point extraction ========
    print("\n" + "=" * 60)
    print("Step 2: SAM3 segmentation + point extraction")
    print("=" * 60)

    from mysf.pipelines.event_centric.sam3_segmenter import Sam3VideoSegmenter
    segmenter = Sam3VideoSegmenter()

    # Parse frustum colors from CLI or use defaults
    if args.bg_cam_color:
        _rgb = [int(x) for x in args.bg_cam_color.split(",")]
        cam_colors = [[_rgb[0]/255, _rgb[1]/255, _rgb[2]/255]]
    else:
        cam_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1],
        ]
    if args.fg_cam_colors:
        fg_cam_colors = []
        for part in args.fg_cam_colors.split(";"):
            _rgb = [int(x) for x in part.strip().split(",")]
            fg_cam_colors.append([_rgb[0]/255, _rgb[1]/255, _rgb[2]/255])
    else:
        fg_cam_colors = [[1, 0.5, 0], [0.5, 0, 1], [0, 0.8, 0.5], [1, 0, 0.5]]

    all_bg_points = []
    all_bg_colors = []
    camera_pose_tracks = []  # list of {"poses": ..., "color": ..., "num_frames": ...}
    all_fg_results = {}  # prompt -> (per_frame_fg_points, per_frame_fg_colors)

    def _load_frame_data(store_map, i):
        """Load frame points/colors/confs from disk for frame i. Returns None tuple if black."""
        si = store_map[i]
        if si < 0:
            return None, None, None
        return pts_store[si], col_store[si], conf_store[si]

    for entry in video_entries:
        all_frames = entry["all_frames"]
        valid_mask = entry["valid_mask"]
        num = entry["num_frames"]  # total frames including black
        store_map = entry["store_map"]
        poses = entry["poses"]

        if entry["is_bg"]:
            vi = entry["video_idx"]
            print(f"\n--- Processing BG {vi}: {entry['path']} ({num} frames, {sum(valid_mask)} valid) ---")

            # SAM3: remove foreground objects (use all frames for temporal consistency)
            print(f"  SAM3: removing foreground from bg...")
            pil_frames = [PILImage.fromarray(f) for f in all_frames]
            bg_fg_masks = None
            for prompt in bg_removal_prompts:
                print(f"    prompt: '{prompt}'")
                masks_i = segmenter.segment(
                    video_path=pil_frames, prompts=[prompt],
                    frame_index=0, expected_frames=num,
                )
                if masks_i.size == 0:
                    print(f"      -> no mask found, skipping")
                    continue
                bg_fg_masks = masks_i if bg_fg_masks is None else (bg_fg_masks | masks_i)
            for img in pil_frames:
                img.close()
            del pil_frames

            if bg_fg_masks is not None:
                _, resized_bg_fg_masks = _align_sam3_masks(bg_fg_masks, num, H_s, W_s, args.mask_dilate)
                del bg_fg_masks
            else:
                resized_bg_fg_masks = np.zeros((num, H_s, W_s), dtype=bool)

            bg_point_mask = entry.get("bg_point_mask", [True] * num)
            _empty = np.empty((0, 3), dtype=np.float32)
            for i in range(num):
                if not valid_mask[i] or not bg_point_mask[i]:
                    all_bg_points.append(_empty)
                    all_bg_colors.append(_empty)
                    continue
                fp, fc, fconf = _load_frame_data(store_map, i)
                if fp is None:
                    all_bg_points.append(_empty)
                    all_bg_colors.append(_empty)
                    continue
                thresh = max(np.percentile(fconf.ravel(), args.conf_percentile), 1e-5)
                conf_mask = fconf >= thresh
                fg_mask = resized_bg_fg_masks[i]
                valid = conf_mask & ~fg_mask
                pts = fp.reshape(-1, 3)[valid.ravel()]
                col = fc.reshape(-1, 3)[valid.ravel()]
                n_removed = (conf_mask & fg_mask).sum()

                if args.bg_voxel_size > 0 and len(pts) > 0:
                    pcd = make_pcd(pts, col)
                    pcd = pcd.voxel_down_sample(voxel_size=args.bg_voxel_size)
                    pts = np.asarray(pcd.points).astype(np.float32)
                    col = np.asarray(pcd.colors).astype(np.float32)

                all_bg_points.append(pts)
                all_bg_colors.append(col)
                if i % 5 == 0 or i == num - 1:
                    print(f"    BG frame {i}: {len(pts):,} pts (removed {n_removed:,} fg pts)")

            del resized_bg_fg_masks

            # Camera pose track for animated visualization (only valid frames)
            if not args.no_camera_viz:
                color = cam_colors[vi % len(cam_colors)]
                valid_poses = poses[np.array(valid_mask)]
                if args.cam_smooth > 0:
                    valid_poses = smooth_poses(valid_poses, window_size=args.cam_smooth)
                    print(f"  Smoothed BG camera poses (window={args.cam_smooth})")
                cam_centers = np.array([-valid_poses[pi][:3, :3].T @ valid_poses[pi][:3, 3] for pi in range(len(valid_poses))])
                if len(cam_centers) > 0:
                    print(f"  Camera centers range: {cam_centers.min(axis=0)} ~ {cam_centers.max(axis=0)}")
                    print(f"  Camera centers mean: {cam_centers.mean(axis=0)}")

                valid_frames_thumb = [_make_thumbnail(f) for f, v in zip(all_frames, valid_mask) if v]
                camera_pose_tracks.append({
                    "poses": valid_poses, "color": color, "num_frames": len(valid_poses),
                    "frames": valid_frames_thumb,
                })
                color_names = ['red','green','blue','yellow','magenta','cyan']
                print(f"  Added camera pose track ({len(valid_poses)} frames, color: {color_names[vi % 6]})")

            # Extract fg objects from bg video if requested
            if args.bg_fg_extract:
                sampled_orig = entry["sampled_orig_indices"]
                # Build per-sampled-frame mask: True if original frame number is in [spec_start, spec_end]
                bg_fg_specs = _parse_bg_fg_extract(args.bg_fg_extract)
                for spec_name, (spec_start, spec_end) in bg_fg_specs:
                    print(f"\n  BG-FG extract: '{spec_name}' orig frames {spec_start}-{spec_end}")
                    in_range = [spec_start <= orig_idx <= spec_end for orig_idx in sampled_orig]
                    n_in_range = sum(1 for ir, vm in zip(in_range, valid_mask) if ir and vm)
                    print(f"    {n_in_range} valid sampled frames in range")

                    # Find first valid frame in range for SAM3 reference
                    first_valid_in_range = next(
                        (i for i in range(num) if in_range[i] and valid_mask[i]), 0
                    )
                    pil_frames_bgfg = [PILImage.fromarray(f) for f in all_frames]
                    bgfg_masks = segmenter.segment(
                        video_path=pil_frames_bgfg, prompts=[spec_name],
                        frame_index=first_valid_in_range, expected_frames=num,
                    )
                    for img in pil_frames_bgfg:
                        img.close()
                    del pil_frames_bgfg

                    if bgfg_masks.size == 0:
                        print(f"    -> no mask found for '{spec_name}', skipping")
                        continue

                    resized_bgfg_masks, _ = _align_sam3_masks(bgfg_masks, num, H_s, W_s, args.mask_dilate)
                    del bgfg_masks

                    _fg_pct = args.fg_conf_percentile
                    per_frame_pts = []
                    per_frame_col = []
                    for i in range(num):
                        if not valid_mask[i] or not in_range[i]:
                            per_frame_pts.append(np.empty((0, 3), dtype=np.float32))
                            per_frame_col.append(np.empty((0, 3), dtype=np.float32))
                            continue
                        fp, fc, fconf = _load_frame_data(store_map, i)
                        if fp is None:
                            per_frame_pts.append(np.empty((0, 3), dtype=np.float32))
                            per_frame_col.append(np.empty((0, 3), dtype=np.float32))
                            continue
                        thresh = max(np.percentile(fconf.ravel(), _fg_pct), 1e-5)
                        fg_valid = (fconf >= thresh) & resized_bgfg_masks[i]
                        pts = fp.reshape(-1, 3)[fg_valid.ravel()]
                        col = fc.reshape(-1, 3)[fg_valid.ravel()]
                        per_frame_pts.append(pts)
                        per_frame_col.append(col)
                        if i % 10 == 0:
                            print(f"    BG-FG frame {i} (orig {sampled_orig[i]}): {len(pts):,} pts")

                    del resized_bgfg_masks
                    bg_fg_key = f"bg_{spec_name}"
                    all_fg_results[bg_fg_key] = (per_frame_pts, per_frame_col)
                    print(f"    Added bg-fg '{spec_name}' ({sum(1 for p in per_frame_pts if len(p) > 0)} non-empty frames)")

        else:
            fi = entry["video_idx"]
            fg_pmt = entry["prompt"]
            print(f"\n--- Processing FG {fi}: {entry['path']} / prompt='{fg_pmt}' ({num} frames, {sum(valid_mask)} valid) ---")

            # SAM3 segmentation (use all frames for temporal consistency)
            # Use first valid (non-black) frame as reference
            first_valid_idx = next((i for i, v in enumerate(valid_mask) if v), 0)
            print(f"  SAM3 segmentation (prompt: '{fg_pmt}', ref frame: {first_valid_idx})...")
            pil_frames = [PILImage.fromarray(f) for f in all_frames]
            fg_masks = segmenter.segment(
                video_path=pil_frames, prompts=[fg_pmt],
                frame_index=first_valid_idx, expected_frames=num,
            )
            for img in pil_frames:
                img.close()
            del pil_frames

            if fg_masks.size == 0:
                print(f"    -> no mask found for '{fg_pmt}', skipping")
                continue

            resized_fg_masks_raw, _ = _align_sam3_masks(fg_masks, num, H_s, W_s, args.mask_dilate)
            del fg_masks

            # Extract per-frame fg points; black frames get empty arrays
            _fg_pct = args.fg_conf_percentile
            per_frame_fg_points = []
            per_frame_fg_colors = []
            for i in range(num):
                if not valid_mask[i]:
                    per_frame_fg_points.append(np.empty((0, 3), dtype=np.float32))
                    per_frame_fg_colors.append(np.empty((0, 3), dtype=np.float32))
                    continue
                fp, fc, fconf = _load_frame_data(store_map, i)
                if fp is None:
                    per_frame_fg_points.append(np.empty((0, 3), dtype=np.float32))
                    per_frame_fg_colors.append(np.empty((0, 3), dtype=np.float32))
                    continue
                thresh = max(np.percentile(fconf.ravel(), _fg_pct), 1e-5)
                fg_valid = (fconf >= thresh) & resized_fg_masks_raw[i]
                pts = fp.reshape(-1, 3)[fg_valid.ravel()]
                col = fc.reshape(-1, 3)[fg_valid.ravel()]
                per_frame_fg_points.append(pts)
                per_frame_fg_colors.append(col)
                if i % 10 == 0 or i == num - 1:
                    print(f"    FG frame {i}: {len(pts):,} pts")

            all_fg_results[fg_pmt] = (per_frame_fg_points, per_frame_fg_colors)

            # FG camera: frustum only, no trail
            if not args.no_camera_viz:
                color = fg_cam_colors[fi % len(fg_cam_colors)]
                valid_poses = poses[np.array(valid_mask)]
                valid_frames_thumb = [_make_thumbnail(f) for f, v in zip(all_frames, valid_mask) if v]
                camera_pose_tracks.append({
                    "poses": valid_poses, "color": color, "num_frames": len(valid_poses),
                    "valid_mask": list(valid_mask),
                    "num_total": num,
                    "no_trail": True,
                    "frames": valid_frames_thumb,
                })
                print(f"  Added fg camera pose track ({len(valid_poses)} valid / {num} total frames, no trail)")

        # Free per-video data
        del entry["all_frames"]  # raw frames no longer needed
        gc.collect()

    del video_entries
    del segmenter
    # Disk stores no longer needed after point extraction
    del pts_store, col_store, conf_store
    gc.collect()

    # Background is kept per-frame for progressive reveal
    total_bg_pts = sum(len(p) for p in all_bg_points)
    print(f"\nTotal background: {total_bg_pts:,} pts across {len(all_bg_points)} frames")

    # ======== Step 3: Save outputs ========
    print("\n" + "=" * 60)
    print("Step 3: Saving outputs")
    print("=" * 60)

    if args.save_glb:
        if total_bg_pts > 0:
            merged_bg_pts = np.concatenate(all_bg_points, axis=0)
            merged_bg_col = np.concatenate(all_bg_colors, axis=0)
            bg_path = os.path.join(args.output_dir, "background.glb")
            save_glb(merged_bg_pts, merged_bg_col, bg_path, sphere_radius=args.sphere_radius)
            print(f"Saved background: {bg_path} ({len(merged_bg_pts):,} pts)")
            del merged_bg_pts, merged_bg_col

        actual_indices = sorted(fg_frame_indices)[::args.sample_rate] if fg_frame_indices else None
        for fg_pmt, (pts_list, col_list) in all_fg_results.items():
            fg_dir = os.path.join(args.output_dir, f"fg_{fg_pmt}")
            os.makedirs(fg_dir, exist_ok=True)
            if actual_indices is None:
                actual_indices = list(range(0, len(pts_list) * args.sample_rate, args.sample_rate))
            for i, (pts, col) in enumerate(zip(pts_list, col_list)):
                if len(pts) > 0:
                    label = actual_indices[i] if i < len(actual_indices) else i
                    fg_path = os.path.join(fg_dir, f"fg_{fg_pmt}_{label:04d}.glb")
                    save_glb(pts, col, fg_path, sphere_radius=args.sphere_radius)
            print(f"Saved {fg_pmt} frames to {fg_dir}")

    # ======== Step 4: Render video ========
    print("\n" + "=" * 60)
    print("Step 4: Rendering 4D world video")
    print("=" * 60)

    if all_fg_results:
        max_frames = max(len(pts_list) for pts_list, _ in all_fg_results.values())
        merged_fg_points = []
        merged_fg_colors = []
        for i in range(max_frames):
            frame_pts = []
            frame_col = []
            for fg_pmt, (pts_list, col_list) in all_fg_results.items():
                if i < len(pts_list) and len(pts_list[i]) > 0:
                    frame_pts.append(pts_list[i])
                    frame_col.append(col_list[i])
            if frame_pts:
                merged_fg_points.append(np.concatenate(frame_pts, axis=0))
                merged_fg_colors.append(np.concatenate(frame_col, axis=0))
            else:
                merged_fg_points.append(np.empty((0, 3)))
                merged_fg_colors.append(np.empty((0, 3)))

        render_world_video(
            output_dir=args.output_dir,
            per_frame_bg_points=all_bg_points,
            per_frame_bg_colors=all_bg_colors,
            per_frame_fg_points=merged_fg_points,
            per_frame_fg_colors=merged_fg_colors,
            camera_pose_tracks=camera_pose_tracks,
            frustum_size=args.frustum_size,
            frustum_z_push=args.frustum_z_push,
            fps=args.render_fps,
            render_size=(args.render_width, args.render_height),
            cam_az=args.cam_az,
            cam_el=args.cam_el,
            cam_dist_scale=args.cam_dist,
            point_size=args.point_size,
            fov=args.fov,
            frustum_image_scale=args.frustum_image_scale,
            frustum_image_brightness=args.frustum_image_brightness,
        )

    # Auto-combine videos if requested
    # Layout: [FG col] [4D render] [BG video]  with title row on top
    #         All three columns have the same height (= render_height)
    # Uses --bg-videos and --fg-video paths automatically
    if args.combine:
        import subprocess
        bg_src = bg_video_list[0]  # first bg video
        fg_srcs = fg_video_list if args.fg_video else []
        render_path = os.path.join(args.output_dir, "4d_world.mp4")
        combined_path = os.path.join(args.output_dir, "4d_world_combined.mp4")
        border = args.combine_border

        # Parse colors
        bg_color_rgb = args.bg_cam_color if args.bg_cam_color else "255,0,0"
        fg_colors_rgb = []
        if args.fg_cam_colors:
            for part in args.fg_cam_colors.split(";"):
                fg_colors_rgb.append(part.strip())
        else:
            fg_colors_rgb = ["255,128,0", "128,0,255"]

        def _rgb_to_hex(rgb_str):
            r, g, b = [int(x) for x in rgb_str.split(",")]
            return f"0x{r:02X}{g:02X}{b:02X}"

        def _even(x):
            """Round to nearest even number."""
            return x if x % 2 == 0 else x + 1

        n_fg = len(fg_srcs)

        # Determine content height: if combine-height is set, reserve space for title
        # Title font size is based on fg column width, but we need to estimate first
        # Use iterative approach: start with content_h, compute font, adjust
        if args.combine_height > 0:
            # First pass estimate: title ~80px
            content_h = _even(args.combine_height - 80)
        else:
            content_h = args.render_height

        # 4D render: scale to fit content_h, maintaining aspect ratio
        render_aspect = args.render_width / args.render_height
        render_w = _even(int(content_h * render_aspect))

        # FG column: videos stacked vertically, total height = content_h
        if n_fg > 0:
            fg_padded_h = _even(content_h // n_fg)
            fg_inner_h = fg_padded_h - 2 * border
            fg_inner_w = _even(int(fg_inner_h * 16 / 9))
            fg_padded_w = fg_inner_w + 2 * border
        else:
            fg_padded_w = 0

        # BG column: height = content_h
        bg_inner_h = _even(content_h - 2 * border)
        bg_inner_w = _even(int(bg_inner_h * 16 / 9))
        bg_padded_w = bg_inner_w + 2 * border
        bg_padded_h = bg_inner_h + 2 * border

        # Build filter
        # inputs: [0]=4d_render, [1]=bg, [2..]=fg videos
        filter_parts = []

        # BG video (input 1) with border -> right column
        filter_parts.append(
            f"[1:v]scale={bg_inner_w}:{bg_inner_h},"
            f"pad={bg_padded_w}:{bg_padded_h}:{border}:{border}:color={_rgb_to_hex(bg_color_rgb)}[bg]"
        )

        # FG videos: preprocess to replace all-black frames with white
        import cv2
        fg_processed_paths = []
        for i, fg_path in enumerate(fg_srcs):
            cap = cv2.VideoCapture(fg_path)
            fps_src = cap.get(cv2.CAP_PROP_FPS)
            w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tmp_path = os.path.join(args.output_dir, f"_fg{i}_white.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, fps_src, (w_src, h_src))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.mean() < 5.0:
                    frame = np.full_like(frame, 255)  # white
                writer.write(frame)
            cap.release()
            writer.release()
            fg_processed_paths.append(tmp_path)
            print(f"  Preprocessed FG{i}: black frames → white → {tmp_path}")

        # FG videos with borders -> middle column
        fg_labels = []
        for i, fg_path in enumerate(fg_processed_paths):
            c = fg_colors_rgb[i] if i < len(fg_colors_rgb) else "255,255,255"
            filter_parts.append(
                f"[{i+2}:v]scale={fg_inner_w}:{fg_inner_h},"
                f"pad={fg_padded_w}:{fg_padded_h}:{border}:{border}:color={_rgb_to_hex(c)}[fg{i}]"
            )
            fg_labels.append(f"[fg{i}]")

        # vstack fg videos -> left column
        if n_fg > 1:
            filter_parts.append("".join(fg_labels) + f"vstack=inputs={n_fg}[fgcol]")
            mid_label = "[fgcol]"
        elif n_fg == 1:
            mid_label = fg_labels[0]
        else:
            mid_label = None

        # Force all columns to exact content_h, then hstack
        filter_parts.append(f"[0:v]scale={render_w}:{content_h}[render]")
        filter_parts.append(f"[bg]scale={bg_padded_w}:{content_h}[bgfit]")

        if mid_label:
            filter_parts.append(f"{mid_label}scale={fg_padded_w}:{content_h}[fgfit]")
            filter_parts.append("[fgfit][render][bgfit]hstack=inputs=3[content]")
        else:
            filter_parts.append("[render][bgfit]hstack=inputs=2[content]")

        # Add white title bar on top via pad, then draw text directly
        fg_col_center = fg_padded_w // 2
        render_center = fg_padded_w + render_w // 2
        bg_col_center = fg_padded_w + render_w + bg_padded_w // 2
        # Font size: scale with smallest column width, capped at 60
        min_col_w = min(w for w in [fg_padded_w, render_w, bg_padded_w] if w > 0)
        font_size = min(60, max(28, int(min_col_w / 10)))
        title_h = _even(font_size + 24)
        font = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
        filter_parts.append(
            f"[content]pad=iw:ih+{title_h}:0:{title_h}:white,"
            f"drawtext=text='Monitors':fontfile={font}:fontsize={font_size}:fontcolor=black:x={fg_col_center}-tw/2:y=({title_h}-th)/2,"
            f"drawtext=text='World States':fontfile={font}:fontsize={font_size}:fontcolor=black:x={render_center}-tw/2:y=({title_h}-th)/2,"
            f"drawtext=text='Renderer':fontfile={font}:fontsize={font_size}:fontcolor=black:x={bg_col_center}-tw/2:y=({title_h}-th)/2"
            f"[out]"
        )

        filter_complex = ";\n".join(filter_parts)

        inputs = ["-i", render_path, "-i", bg_src]
        for fg_path in fg_processed_paths:
            inputs.extend(["-i", fg_path])

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-r", str(int(args.render_fps)),
            "-shortest",
            combined_path,
        ]
        print(f"\n{'='*60}")
        print(f"Combining videos → {combined_path}")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)
        print(f"  Combined video saved: {combined_path}")
        # Cleanup temp preprocessed FG videos
        for tmp in fg_processed_paths:
            if os.path.exists(tmp):
                os.remove(tmp)

    # Cleanup temp dir
    import shutil
    shutil.rmtree(_tmpdir, ignore_errors=True)
    print(f"  Cleaned up temp dir: {_tmpdir}")

    print(f"\nDone! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
