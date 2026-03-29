"""4D point cloud pipeline: STream3R + SAM3 foreground/background separation.

Reconstructs per-frame 3D point clouds with STream3R, segments foreground using
SAM3 text prompts, merges all background, and outputs per-frame PLY files
(merged background + that frame's foreground).

Supports two modes:
  - Single video: one video, SAM3 splits foreground/background.
  - Two videos: separate --bg-video and --fg-video, both processed through the
    same STream3R session (shared world coordinates). SAM3 runs on --fg-video only.

Usage:
    # Single video mode
    python misc/STream3R/infer_stream3r_4d.py video.mp4 --fg-prompt "person" -o outputs/4d/

    # Two-video mode
    python misc/STream3R/infer_stream3r_4d.py --bg-video bg.mp4 --fg-video fg.mp4 --fg-prompt "person"
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image as PILImage

# Ensure project root and STream3R dir are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STREAM3R_DIR = os.path.dirname(os.path.abspath(__file__))
if STREAM3R_DIR not in sys.path:
    sys.path.insert(0, STREAM3R_DIR)

import re

from infer_stream3r import (
    extract_frames_from_video,
    frames_to_preprocessed_tensor,
    parse_frame_indices,
)
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession


def parse_paired_frames(spec: str) -> list[tuple[int, int]] | None:
    """Parse paired frame spec like '[0,0]--[11,11]--[37,25]'.

    Returns list of (fg_frame, bg_frame) tuples, or None if not paired format.
    """
    if "[" not in spec:
        return None
    pairs = []
    for part in spec.split("--"):
        part = part.strip()
        m = re.match(r"\[(\d+)\s*,\s*(\d+)\]", part)
        if not m:
            return None
        pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def make_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """Build an Open3D PointCloud with color normalization."""
    pcd = o3d.geometry.PointCloud()
    if len(points) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    cols = colors.copy().astype(np.float64)
    if cols.size > 0 and cols.max() > 1.5:
        cols = cols / 255.0
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    return pcd


def save_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
    pcd = make_pcd(points, colors)
    o3d.io.write_point_cloud(path, pcd)


def _build_sphere_mesh(points: np.ndarray, cols_u8: np.ndarray, tv: np.ndarray, tf: np.ndarray) -> trimesh.Trimesh:
    """Build a single Trimesh of icospheres for a batch of points."""
    Vt, Ft = len(tv), len(tf)
    N = len(points)
    all_verts = (np.repeat(tv[None], N, axis=0) + points[:, None, :]).reshape(-1, 3)
    offsets = (np.arange(N) * Vt)[:, None, None]
    all_faces = (tf[None] + offsets).reshape(-1, 3)
    rgba = np.concatenate([cols_u8, np.full((N, 1), 255, dtype=np.uint8)], axis=1)
    all_colors = np.repeat(rgba, Vt, axis=0)
    return trimesh.Trimesh(vertices=all_verts, faces=all_faces, vertex_colors=all_colors, process=False)


def save_glb(points: np.ndarray, colors: np.ndarray, path: str, sphere_radius: float = 0.005) -> None:
    """Save a colored point cloud as GLB with each point as a small icosphere."""
    if len(points) == 0:
        trimesh.Scene().export(path)
        return
    cols = colors.copy()
    if cols.max() <= 1.5:
        cols = (cols * 255).clip(0, 255).astype(np.uint8)
    else:
        cols = cols.clip(0, 255).astype(np.uint8)

    template = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
    tv = np.asarray(template.vertices)  # (Vt, 3)
    tf = np.asarray(template.faces)     # (Ft, 3)
    Vt = len(tv)
    N = len(points)

    # GLB uses uint32 for buffer offsets/lengths; keep total mesh data well under 2^32 bytes.
    # Each point creates Vt vertices (12 bytes pos + 4 bytes color) + Ft faces (12 bytes).
    # Limit to ~500K points per chunk to stay safe.
    max_points_per_chunk = min(500_000, (2**32 - 1) // Vt)
    if N <= max_points_per_chunk:
        mesh = _build_sphere_mesh(points, cols, tv, tf)
        mesh.export(path)
    else:
        # Split into chunks, export as scene
        scene = trimesh.Scene()
        for ci, start in enumerate(range(0, N, max_points_per_chunk)):
            end = min(start + max_points_per_chunk, N)
            chunk_mesh = _build_sphere_mesh(points[start:end], cols[start:end], tv, tf)
            scene.add_geometry(chunk_mesh, node_name=f"chunk_{ci}")
        scene.export(path)


def save_4d_glbs(
    output_dir: str,
    per_frame_bg_points: list[np.ndarray],
    per_frame_bg_colors: list[np.ndarray],
    per_frame_fg_points: list[np.ndarray],
    per_frame_fg_colors: list[np.ndarray],
    sphere_radius: float = 0.005,
    fg_frame_indices: list[int] | None = None,
) -> None:
    """Save per-frame GLB files with progressive background accumulation.

    If fg_frame_indices is provided, GLB filenames use actual video frame numbers.
    """
    num_frames = len(per_frame_fg_points)
    glb_dir = os.path.join(output_dir, "render_glb")
    os.makedirs(glb_dir, exist_ok=True)

    print(f"\n=== Saving {num_frames} GLB frames (progressive bg) ===")

    accum_bg_pts = []
    accum_bg_col = []

    for i in range(num_frames):
        # Progressive accumulation: always accumulate bg up to current frame
        if len(per_frame_bg_points[i]) > 0:
            accum_bg_pts.append(per_frame_bg_points[i])
            accum_bg_col.append(per_frame_bg_colors[i])
        cur_bg_pts = np.concatenate(accum_bg_pts, axis=0) if accum_bg_pts else np.empty((0, 3))
        cur_bg_col = np.concatenate(accum_bg_col, axis=0) if accum_bg_pts else np.empty((0, 3))

        combined_pts = np.concatenate([cur_bg_pts, per_frame_fg_points[i]], axis=0)
        combined_col = np.concatenate([cur_bg_col, per_frame_fg_colors[i]], axis=0)

        label = fg_frame_indices[i] if fg_frame_indices is not None else i
        save_glb(combined_pts, combined_col, os.path.join(glb_dir, f"frame_{label:04d}.glb"), sphere_radius=sphere_radius)

        if i % 5 == 0 or i == num_frames - 1:
            bg_n = len(cur_bg_pts)
            print(f"  frame {label} ({i+1}/{num_frames}): {len(combined_pts):,} pts (bg: {bg_n:,})")

    print(f"Saved {num_frames} GLBs to {glb_dir}")


def save_combined_glb(
    output_dir: str,
    per_frame_bg_points: list[np.ndarray],
    per_frame_bg_colors: list[np.ndarray],
    per_frame_fg_points: list[np.ndarray],
    per_frame_fg_colors: list[np.ndarray],
    sphere_radius: float = 0.005,
    save_frames: list[int] | None = None,
    fg_frame_indices: list[int] | None = None,
    save_bg: bool = True,
    per_prompt_fg_points: dict[str, list[np.ndarray]] | None = None,
    per_prompt_fg_colors: dict[str, list[np.ndarray]] | None = None,
) -> None:
    """Save per-frame foreground GLBs + optionally one accumulated background GLB."""
    print("\n=== Saving per-frame foreground GLBs ===")

    fg_dir = os.path.join(output_dir, "fg_frames")
    os.makedirs(fg_dir, exist_ok=True)
    labels = fg_frame_indices if fg_frame_indices else list(range(len(per_frame_fg_points)))

    # Save per-prompt foreground GLBs if multiple prompts
    if per_prompt_fg_points and len(per_prompt_fg_points) > 1:
        for prompt, pts_list in per_prompt_fg_points.items():
            col_list = per_prompt_fg_colors[prompt]
            for i, (pts, col) in enumerate(zip(pts_list, col_list)):
                if len(pts) > 0:
                    label = labels[i] if i < len(labels) else i
                    fg_path = os.path.join(fg_dir, f"fg_{prompt}_{label:04d}.glb")
                    save_glb(pts, col, fg_path, sphere_radius=sphere_radius)
                    print(f"  Saved fg_{prompt} frame {label:04d}: {fg_path} ({len(pts):,} pts)")
    else:
        # Single prompt — save without prompt name prefix
        for i, (pts, col) in enumerate(zip(per_frame_fg_points, per_frame_fg_colors)):
            if len(pts) > 0:
                label = labels[i] if i < len(labels) else i
                fg_path = os.path.join(fg_dir, f"fg_{label:04d}.glb")
                save_glb(pts, col, fg_path, sphere_radius=sphere_radius)
                print(f"  Saved fg frame {label:04d}: {fg_path} ({len(pts):,} pts)")

    # Save one accumulated background GLB (all bg frames merged)
    if not save_bg:
        print("  Skipping background GLB (--no-save-bg)")
        return
    bg_pts_list = [p for p in per_frame_bg_points if len(p) > 0]
    bg_col_list = [c for c in per_frame_bg_colors if len(c) > 0]
    if bg_pts_list:
        merged_bg_pts = np.concatenate(bg_pts_list, axis=0)
        merged_bg_col = np.concatenate(bg_col_list, axis=0)
        bg_path = os.path.join(output_dir, "background_only.glb")
        save_glb(merged_bg_pts, merged_bg_col, bg_path, sphere_radius=sphere_radius)
        print(f"Saved background GLB: {bg_path} ({len(merged_bg_pts):,} pts)")


def render_4d_video(
    output_dir: str,
    video_path: str,
    per_frame_bg_points: list[np.ndarray],
    per_frame_bg_colors: list[np.ndarray],
    per_frame_fg_points: list[np.ndarray],
    per_frame_fg_colors: list[np.ndarray],
    fps: float = 8.0,
    render_size: tuple[int, int] = (960, 540),
    cam_az: float = 0.0,
    cam_el: float = 20.0,
    cam_dist_scale: float = 0.8,
    point_size: float = 2.0,
    fov: float = 60.0,
    bg_color: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> str:
    """Render per-frame 4D point clouds to an mp4 video using Open3D offscreen.

    Background accumulates progressively: frame i shows bg from frames 0..i + fg from frame i.

    Returns:
        Path to the saved video file.
    """
    num_frames = len(per_frame_fg_points)
    W, H = render_size

    # Build the full scene once to compute bounding box and camera
    all_pts = per_frame_bg_points + per_frame_fg_points
    scene_pts = np.concatenate([p for p in all_pts if len(p) > 0], axis=0)

    # Robust scene statistics: filter outliers via percentile clipping
    lo = np.percentile(scene_pts, 2, axis=0)
    hi = np.percentile(scene_pts, 98, axis=0)
    inlier_mask = np.all((scene_pts >= lo) & (scene_pts <= hi), axis=1)
    inlier_pts = scene_pts[inlier_mask] if inlier_mask.sum() > 100 else scene_pts
    scene_center = np.median(inlier_pts, axis=0)
    scene_extent = np.percentile(
        np.linalg.norm(inlier_pts - scene_center, axis=1), 90
    )

    # Camera: look at scene center from configurable angle
    az = np.radians(cam_az)
    el = np.radians(cam_el)
    cam_dist = scene_extent * cam_dist_scale
    cam_pos = scene_center + cam_dist * np.array([
        np.sin(az) * np.cos(el),
        -np.sin(el),
        -np.cos(az) * np.cos(el),
    ])

    # Build lookat extrinsic (OpenCV convention: X-right, Y-down, Z-forward)
    forward = scene_center - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Extrinsic: world-to-camera rotation + translation
    R = np.stack([right, -up, forward], axis=0)
    t = -R @ cam_pos
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    # Intrinsic: simple pinhole with configurable FOV
    fy = H / (2 * np.tan(np.radians(fov / 2)))
    fx = fy
    cx, cy = W / 2.0, H / 2.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # Setup offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(np.array([*bg_color, 1.0]))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    # Pipe rendered frames directly to ffmpeg (no temp PNGs on disk)
    video_out = os.path.join(output_dir, "4d_render.mp4")

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{W}x{H}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            video_out,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"\n=== Rendering {num_frames} frames to video ===")
    print(f"  Resolution: {W}x{H}, FPS: {fps}")
    print(f"  Camera: az={np.degrees(az):.0f}, el={np.degrees(el):.0f}, dist={cam_dist:.2f}, fov={fov:.0f}")

    # Progressively accumulate background
    accum_bg_pts = []
    accum_bg_col = []

    for i in range(num_frames):
        # Accumulate background up to current frame
        if len(per_frame_bg_points[i]) > 0:
            accum_bg_pts.append(per_frame_bg_points[i])
            accum_bg_col.append(per_frame_bg_colors[i])

        # Rebuild background geometry with accumulated points
        if renderer.scene.has_geometry("background"):
            renderer.scene.remove_geometry("background")
        if accum_bg_pts:
            bg_pcd = make_pcd(np.concatenate(accum_bg_pts, axis=0), np.concatenate(accum_bg_col, axis=0))
        else:
            bg_pcd = make_pcd(np.empty((0, 3)), np.empty((0, 3)))
        renderer.scene.add_geometry("background", bg_pcd, mat)

        # Update foreground
        if renderer.scene.has_geometry("foreground"):
            renderer.scene.remove_geometry("foreground")
        fg_pcd = make_pcd(per_frame_fg_points[i], per_frame_fg_colors[i])
        renderer.scene.add_geometry("foreground", fg_pcd, mat)

        renderer.setup_camera(intrinsic, extrinsic)
        img = np.asarray(renderer.render_to_image())  # RGB uint8 (H, W, 3)
        ffmpeg_proc.stdin.write(img.tobytes())

        if i % 5 == 0 or i == num_frames - 1:
            bg_total = sum(len(p) for p in accum_bg_pts) if accum_bg_pts else 0
            print(f"  Rendered frame {i}/{num_frames-1} (bg: {bg_total:,} pts)")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    print(f"Saved video: {video_out}")
    return video_out


def _stream3r_inference(
    images: torch.Tensor,
    model_path: str,
    device: str,
    mode: str = "causal",
    window_size: int = 5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run STream3R on preprocessed images and return per-frame points/colors/confs."""
    model = STream3R.from_pretrained(model_path).to(device)
    model.eval()
    session = StreamSession(model, mode=mode, window_size=window_size)
    print(f"  Mode: {mode}" + (f", window_size: {window_size}" if mode == "window" else ""))

    frame_points = []
    frame_colors = []
    frame_confs = []
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
            if wp.ndim == 4:
                wp = wp[-1]
            conf = preds.get("world_points_conf", np.ones(wp.shape[:-1]))
            if conf.ndim >= 2 and conf.shape[0] == i + 1:
                conf = conf[-1]
            img = preds["images"]
            if img.ndim == 4 and img.shape[0] == i + 1:
                img = img[-1]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            frame_points.append(wp)
            frame_colors.append(img)
            frame_confs.append(conf)
            print(f"  Frame {i}/{n-1}: points {wp.shape}")

    session.clear()
    del model, session
    torch.cuda.empty_cache()

    return frame_points, frame_colors, frame_confs


def _align_sam3_masks(
    fg_masks: np.ndarray,
    num_frames: int,
    H_s: int,
    W_s: int,
    mask_dilate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align SAM3 masks to STream3R resolution.

    Returns:
        (raw_masks, dilated_masks): raw for foreground extraction, dilated for background removal.
        When mask_dilate <= 0, both are identical.
    """
    if fg_masks.size > 0 and fg_masks.shape[0] == num_frames:
        H_orig, W_orig = fg_masks.shape[1], fg_masks.shape[2]
        new_w = 518
        new_h = round(H_orig * (518 / W_orig) / 14) * 14
        crop_y = 0
        final_h = new_h
        if new_h > 518:
            crop_y = (new_h - 518) // 2
            final_h = 518
        print(f"SAM3 mask: {H_orig}x{W_orig} -> resize to {new_h}x{new_w} -> crop to {final_h}x{new_w}")
        print(f"STream3R output: {H_s}x{W_s}")

        raw_masks = np.zeros((num_frames, H_s, W_s), dtype=bool)
        for t in range(num_frames):
            mask_u8 = fg_masks[t].astype(np.uint8) * 255
            mask_u8 = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_u8 = mask_u8[crop_y:crop_y + final_h, :]
            if mask_u8.shape != (H_s, W_s):
                mask_u8 = cv2.resize(mask_u8, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
            raw_masks[t] = mask_u8 > 0

        if mask_dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * mask_dilate + 1, 2 * mask_dilate + 1)
            )
            dilated_masks = np.zeros_like(raw_masks)
            for t in range(num_frames):
                dilated_masks[t] = cv2.dilate(
                    raw_masks[t].astype(np.uint8), kernel
                ).astype(bool)
            print(f"Dilated masks by {mask_dilate} pixels (for bg removal only)")
        else:
            dilated_masks = raw_masks

        fg_count = sum(raw_masks[t].sum() for t in range(num_frames))
        print(f"Total foreground pixels across all frames: {fg_count:,}")
    else:
        print("SAM3 found no foreground. All points treated as background.")
        raw_masks = np.zeros((num_frames, H_s, W_s), dtype=bool)
        dilated_masks = raw_masks

    return raw_masks, dilated_masks


def run_stream3r_4d(
    video_path: str | None = None,
    bg_video_path: str | None = None,
    fg_video_path: str | None = None,
    output_dir: str = "outputs/4d",
    fg_prompt: str = "",
    sample_rate: int = 1,
    frame_indices: list[int] | None = None,
    bg_sample_rate: int | None = None,
    bg_frame_indices: list[int] | None = None,
    conf_percentile: float = 30.0,
    fg_conf_percentile: float | None = None,
    model_path: str = "hf_cache/yslan--STream3R",
    stream_mode: str = "causal",
    window_size: int = 5,
    sam3_checkpoint: str | None = None,
    device: str = "cuda",
    render_video: bool = True,
    render_fps: float = 8.0,
    cam_az: float = 0.0,
    cam_el: float = 20.0,
    cam_dist_scale: float = 0.8,
    point_size: float = 2.0,
    fov: float = 60.0,
    mask_dilate: int = 3,
    bg_voxel_size: float = 0,
    save_ply_files: bool = True,
    save_render_glb: bool = False,
    save_combined_glb_flag: bool = False,
    sphere_radius: float = 0.005,
    save_frames: list[int] | None = None,
    save_bg: bool = True,
) -> None:
    # Determine mode: single-video vs two-video
    two_video_mode = bg_video_path is not None and fg_video_path is not None
    if not two_video_mode:
        if video_path is None:
            raise ValueError("Must provide either a single video or both --bg-video and --fg-video")

    # If save_frames is set, use it as default frame range for both fg and bg
    if save_frames:
        if frame_indices is None:
            frame_indices = save_frames
        if bg_frame_indices is None:
            bg_frame_indices = save_frames

    # ---- Phase 1: Frame extraction ----
    print("=== Phase 1: Extracting frames ===")

    def _compute_actual_indices(vid_path, sr, f_indices):
        """Compute the actual video frame indices that extract_frames_from_video returns."""
        if f_indices is not None:
            return sorted(f_indices)[::sr]
        cap = cv2.VideoCapture(vid_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return list(range(0, total, sr))

    if two_video_mode:
        _bg_sr = bg_sample_rate if bg_sample_rate is not None else sample_rate
        bg_frames = extract_frames_from_video(bg_video_path, sample_rate=_bg_sr, frame_indices=bg_frame_indices)
        fg_frames = extract_frames_from_video(fg_video_path, sample_rate=sample_rate, frame_indices=frame_indices)
        fg_actual_indices = _compute_actual_indices(fg_video_path, sample_rate, frame_indices)
        num_bg = len(bg_frames)
        num_fg = len(fg_frames)
        print(f"Background: {num_bg} frames from {bg_video_path}")
        print(f"Foreground: {num_fg} frames from {fg_video_path}")
        if num_bg == 0 or num_fg == 0:
            print("No frames extracted. Exiting.")
            return
        # Concatenate: bg first, then fg — all go through same session
        all_frames = bg_frames + fg_frames
    else:
        all_frames = extract_frames_from_video(video_path, sample_rate=sample_rate, frame_indices=frame_indices)
        fg_actual_indices = _compute_actual_indices(video_path, sample_rate, frame_indices)
        num_bg = 0
        num_fg = len(all_frames)
        if num_fg == 0:
            print("No frames extracted. Exiting.")
            return
        fg_frames = all_frames

    # Trim to actual extracted count (video may be shorter than requested range)
    fg_actual_indices = fg_actual_indices[:num_fg]

    num_total = len(all_frames)
    print(f"Total frames for STream3R: {num_total}")

    # ---- Phase 2: STream3R inference (shared session for all frames) ----
    print("\n=== Phase 2: STream3R inference ===")
    images = frames_to_preprocessed_tensor(all_frames, device=device)
    print(f"Input tensor: {images.shape}")

    frame_points, frame_colors, frame_confs = _stream3r_inference(
        images, model_path, device, mode=stream_mode, window_size=window_size
    )
    del images
    torch.cuda.empty_cache()

    H_s, W_s = frame_points[0].shape[:2]
    print(f"STream3R output resolution: {H_s}x{W_s}")

    if two_video_mode:
        # Split results: first num_bg are background, rest are foreground
        bg_points = frame_points[:num_bg]
        bg_colors = frame_colors[:num_bg]
        bg_confs = frame_confs[:num_bg]
        fg_points_list = frame_points[num_bg:]
        fg_colors_list = frame_colors[num_bg:]
        fg_confs_list = frame_confs[num_bg:]
        del frame_points, frame_colors, frame_confs
    else:
        fg_points_list = frame_points
        fg_colors_list = frame_colors
        fg_confs_list = frame_confs

    # ---- Phase 3: SAM3 segmentation ----
    # Support multiple prompts separated by commas
    prompt_list = [p.strip() for p in fg_prompt.split(",") if p.strip()]
    print(f"\n=== Phase 3: SAM3 segmentation (prompts: {prompt_list}) ===")
    from mysf.pipelines.event_centric.sam3_segmenter import Sam3VideoSegmenter

    segmenter = Sam3VideoSegmenter(checkpoint_path=sam3_checkpoint)

    # Segment foreground video — per-prompt masks + union
    print(f"  Segmenting fg video ({num_fg} frames)...")
    pil_fg = [PILImage.fromarray(f) for f in fg_frames]
    fg_masks = None
    per_prompt_fg_masks_raw = {}  # prompt -> resized raw masks
    for prompt in prompt_list:
        print(f"    prompt: '{prompt}'")
        masks_i = segmenter.segment(
            video_path=pil_fg,
            prompts=[prompt],
            frame_index=0,
            expected_frames=num_fg,
        )
        if masks_i.size == 0:
            print(f"      -> no mask found, skipping")
            continue
        raw_i, _ = _align_sam3_masks(masks_i, num_fg, H_s, W_s, mask_dilate)
        per_prompt_fg_masks_raw[prompt] = raw_i
        fg_masks = masks_i if fg_masks is None else (fg_masks | masks_i)
    for img in pil_fg:
        img.close()
    del pil_fg

    if fg_masks is None:
        print("  WARNING: No foreground found for any prompt. All points treated as background.")
        fg_masks = np.zeros((num_fg, fg_frames[0].shape[0], fg_frames[0].shape[1]), dtype=bool)
    resized_fg_masks_raw, resized_fg_masks_dilated = _align_sam3_masks(fg_masks, num_fg, H_s, W_s, mask_dilate)
    del fg_masks

    # Segment background video to remove foreground from it
    if two_video_mode:
        print(f"  Segmenting bg video ({num_bg} frames) to remove foreground...")
        pil_bg = [PILImage.fromarray(f) for f in bg_frames]
        bg_fg_masks = None
        for prompt in prompt_list:
            print(f"    prompt: '{prompt}'")
            masks_i = segmenter.segment(
                video_path=pil_bg,
                prompts=[prompt],
                frame_index=0,
                expected_frames=num_bg,
            )
            if masks_i.size == 0:
                print(f"      -> no mask found, skipping")
                continue
            bg_fg_masks = masks_i if bg_fg_masks is None else (bg_fg_masks | masks_i)
        for img in pil_bg:
            img.close()
        del pil_bg

        if bg_fg_masks is None:
            bg_fg_masks = np.zeros((num_bg, bg_frames[0].shape[0], bg_frames[0].shape[1]), dtype=bool)
        _, resized_bg_fg_masks = _align_sam3_masks(bg_fg_masks, num_bg, H_s, W_s, mask_dilate)
        del bg_fg_masks
    else:
        resized_bg_fg_masks = None

    del segmenter
    del all_frames, fg_frames
    if two_video_mode:
        del bg_frames
    torch.cuda.empty_cache()

    # ---- Phase 4: Split, merge, save ----
    print(f"\n=== Phase 4: Splitting and saving to {output_dir} ===")
    os.makedirs(output_dir, exist_ok=True)

    # Process background frames (per-frame lists for progressive accumulation)
    def _voxel_downsample(pts, cols):
        """Optionally voxel-downsample background points."""
        if bg_voxel_size <= 0 or len(pts) == 0:
            return pts, cols
        pcd = make_pcd(pts, cols)
        pcd = pcd.voxel_down_sample(voxel_size=bg_voxel_size)
        ds_pts = np.asarray(pcd.points).astype(pts.dtype)
        ds_cols = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8) if cols.max() > 1.5 else np.asarray(pcd.colors)
        return ds_pts, ds_cols

    bg_frame_points = []  # per bg-source-frame points
    bg_frame_colors = []

    if two_video_mode:
        # Background from bg_video, with foreground removed via SAM3
        for i in range(num_bg):
            wp = bg_points[i]
            col = bg_colors[i]
            conf = bg_confs[i]
            thresh = max(np.percentile(conf.ravel(), conf_percentile), 1e-5)
            conf_mask = conf >= thresh
            bg_fg_mask = resized_bg_fg_masks[i]
            valid = conf_mask & ~bg_fg_mask
            pts_flat = wp.reshape(-1, 3)
            col_flat = col.reshape(-1, 3)
            bg_pts = pts_flat[valid.ravel()]
            bg_col = col_flat[valid.ravel()]
            n_before = len(bg_pts)
            bg_pts, bg_col = _voxel_downsample(bg_pts, bg_col)
            bg_frame_points.append(bg_pts)
            bg_frame_colors.append(bg_col)
            ds_info = f" -> {len(bg_pts):,} after voxel ds" if bg_voxel_size > 0 else ""
            print(f"  BG frame {i:04d}: {n_before:,} pts{ds_info} (removed {(conf_mask & bg_fg_mask).sum():,} fg pts)")
        del bg_points, bg_colors, bg_confs, resized_bg_fg_masks

    per_frame_fg_points = []
    per_frame_fg_colors = []
    # Per-prompt foreground points: {prompt: [per_frame_pts, ...]}
    per_prompt_fg_points = {p: [] for p in per_prompt_fg_masks_raw}
    per_prompt_fg_colors = {p: [] for p in per_prompt_fg_masks_raw}

    for i in range(num_fg):
        wp = fg_points_list[i]
        col = fg_colors_list[i]
        conf = fg_confs_list[i]
        fg_mask_raw = resized_fg_masks_raw[i]
        fg_mask_dilated = resized_fg_masks_dilated[i]

        _fg_pct = fg_conf_percentile if fg_conf_percentile is not None else conf_percentile
        fg_thresh = max(np.percentile(conf.ravel(), _fg_pct), 1e-5)
        bg_thresh = max(np.percentile(conf.ravel(), conf_percentile), 1e-5)
        fg_conf_mask = conf >= fg_thresh
        bg_conf_mask = conf >= bg_thresh

        # Foreground: use raw (tight) mask
        fg_valid = fg_conf_mask & fg_mask_raw
        # Background: use dilated mask to ensure clean separation
        bg_valid = bg_conf_mask & ~fg_mask_dilated

        pts_flat = wp.reshape(-1, 3)
        col_flat = col.reshape(-1, 3)

        fg_pts = pts_flat[fg_valid.ravel()]
        fg_col = col_flat[fg_valid.ravel()]
        bg_pts = pts_flat[bg_valid.ravel()]
        bg_col = col_flat[bg_valid.ravel()]

        per_frame_fg_points.append(fg_pts)
        per_frame_fg_colors.append(fg_col)

        # Per-prompt foreground extraction
        for prompt, prompt_masks in per_prompt_fg_masks_raw.items():
            p_valid = fg_conf_mask & prompt_masks[i]
            per_prompt_fg_points[prompt].append(pts_flat[p_valid.ravel()])
            per_prompt_fg_colors[prompt].append(col_flat[p_valid.ravel()])

        if not two_video_mode:
            bg_pts, bg_col = _voxel_downsample(bg_pts, bg_col)
            bg_frame_points.append(bg_pts)
            bg_frame_colors.append(bg_col)

        print(f"  FG frame {i:04d}: {fg_pts.shape[0]:,} fg pts, {bg_pts.shape[0]:,} bg pts")

    del fg_points_list, fg_colors_list, fg_confs_list, resized_fg_masks_raw, resized_fg_masks_dilated

    # Build per-fg-frame background lists (synchronized with fg frames).
    # Each entry = new bg points to add at that fg frame.
    # In two-video mode, spread bg frames evenly across fg frames.
    empty = np.empty((0, 3))
    per_fg_frame_bg_points = [empty] * num_fg
    per_fg_frame_bg_colors = [empty] * num_fg

    if two_video_mode:
        # Distribute num_bg bg frames across num_fg fg frames
        for b in range(len(bg_frame_points)):
            fg_idx = int(b * num_fg / len(bg_frame_points))
            fg_idx = min(fg_idx, num_fg - 1)
            if len(per_fg_frame_bg_points[fg_idx]) == 0:
                per_fg_frame_bg_points[fg_idx] = bg_frame_points[b]
                per_fg_frame_bg_colors[fg_idx] = bg_frame_colors[b]
            else:
                per_fg_frame_bg_points[fg_idx] = np.concatenate(
                    [per_fg_frame_bg_points[fg_idx], bg_frame_points[b]], axis=0
                )
                per_fg_frame_bg_colors[fg_idx] = np.concatenate(
                    [per_fg_frame_bg_colors[fg_idx], bg_frame_colors[b]], axis=0
                )
        del bg_frame_points, bg_frame_colors
    else:
        per_fg_frame_bg_points = bg_frame_points
        per_fg_frame_bg_colors = bg_frame_colors

    if save_ply_files:
        # For PLY, save with fully merged background
        merged_bg_pts = np.concatenate(per_fg_frame_bg_points, axis=0)
        merged_bg_col = np.concatenate(per_fg_frame_bg_colors, axis=0)

        bg_path = os.path.join(output_dir, "background.ply")
        save_ply(merged_bg_pts, merged_bg_col, bg_path)
        print(f"\nSaved background.ply ({merged_bg_pts.shape[0]:,} pts)")

        for i in range(num_fg):
            combined_pts = np.concatenate([merged_bg_pts, per_frame_fg_points[i]], axis=0)
            combined_col = np.concatenate([merged_bg_col, per_frame_fg_colors[i]], axis=0)
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.ply")
            save_ply(combined_pts, combined_col, frame_path)
            print(f"  Saved frame_{i:04d}.ply ({combined_pts.shape[0]:,} pts)")
        del merged_bg_pts, merged_bg_col
    else:
        print(f"\nSkipped PLY saving (--no-ply)")

    # ---- Phase 5: Save GLBs / Render video ----
    if save_render_glb:
        save_4d_glbs(
            output_dir=output_dir,
            per_frame_bg_points=per_fg_frame_bg_points,
            per_frame_bg_colors=per_fg_frame_bg_colors,
            per_frame_fg_points=per_frame_fg_points,
            per_frame_fg_colors=per_frame_fg_colors,
            sphere_radius=sphere_radius,
            fg_frame_indices=fg_actual_indices,
        )

    if save_combined_glb_flag:
        save_combined_glb(
            output_dir=output_dir,
            per_frame_bg_points=per_fg_frame_bg_points,
            per_frame_bg_colors=per_fg_frame_bg_colors,
            per_frame_fg_points=per_frame_fg_points,
            per_frame_fg_colors=per_frame_fg_colors,
            sphere_radius=sphere_radius,
            save_frames=save_frames,
            fg_frame_indices=fg_actual_indices,
            save_bg=save_bg,
            per_prompt_fg_points=per_prompt_fg_points,
            per_prompt_fg_colors=per_prompt_fg_colors,
        )

    if render_video:
        _vid_path = fg_video_path if two_video_mode else video_path
        render_4d_video(
            output_dir=output_dir,
            video_path=_vid_path,
            per_frame_bg_points=per_fg_frame_bg_points,
            per_frame_bg_colors=per_fg_frame_bg_colors,
            per_frame_fg_points=per_frame_fg_points,
            per_frame_fg_colors=per_frame_fg_colors,
            fps=render_fps,
            cam_az=cam_az,
            cam_el=cam_el,
            cam_dist_scale=cam_dist_scale,
            point_size=point_size,
            fov=fov,
        )

    print(f"\nDone! {num_fg} fg frames + background saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="4D point cloud: STream3R + SAM3 foreground/background separation"
    )
    # Input: either a single positional video, or --bg-video + --fg-video
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to input video (single-video mode)")
    parser.add_argument("--bg-video", type=str, default=None,
                        help="Background video (two-video mode, shared STream3R session with --fg-video)")
    parser.add_argument("--fg-video", type=str, default=None,
                        help="Foreground video (two-video mode, SAM3 runs on this)")
    parser.add_argument("--fg-prompt", required=True, help="Text prompt for foreground (e.g. 'person')")
    parser.add_argument("--output-dir", "-o", default="outputs/4d", help="Output directory for PLY files")
    parser.add_argument("--sample-rate", type=int, default=1, help="Keep every N-th frame (applies to fg video)")
    parser.add_argument("--frames", type=str, default=None,
                        help="Frame selection. Normal: '0,1,2,3-10,15'. "
                             "Paired: '[fg,bg]--[fg,bg]--...' e.g. '[0,0]--[11,11]--[37,25]'")
    parser.add_argument("--bg-sample-rate", type=int, default=None,
                        help="Sample rate for bg video (default: same as --sample-rate)")
    parser.add_argument("--bg-frames", type=str, default=None,
                        help="Select specific frames for bg video")
    parser.add_argument("--conf-percentile", type=float, default=30.0,
                        help="Confidence percentile threshold for background (default: 30, keeps top 70%%)")
    parser.add_argument("--fg-conf-percentile", type=float, default=None,
                        help="Confidence percentile threshold for foreground (default: same as --conf-percentile)")
    parser.add_argument("--model-path", type=str, default="hf_cache/yslan--STream3R")
    parser.add_argument("--stream-mode", type=str, default="causal", choices=["causal", "window"],
                        help="STream3R inference mode (default: causal)")
    parser.add_argument("--window-size", type=int, default=5,
                        help="Window size for window mode (default: 5)")
    parser.add_argument("--sam3-checkpoint", type=str, default=None,
                        help="Path to SAM3 checkpoint (default: use SAM3 default)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering")
    parser.add_argument("--render-fps", type=float, default=8.0, help="Render video FPS (default: 8)")
    parser.add_argument("--cam-az", type=float, default=0.0, help="Camera azimuth in degrees (default: 0, front view)")
    parser.add_argument("--cam-el", type=float, default=20.0, help="Camera elevation in degrees (default: 20, slightly overhead)")
    parser.add_argument("--cam-dist", type=float, default=0.8, help="Camera distance scale (default: 0.8, smaller=closer)")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size for rendering (default: 2.0)")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera field of view in degrees (default: 60, smaller=narrower/zoomed in)")
    parser.add_argument("--mask-dilate", type=int, default=3, help="Dilate foreground mask by N pixels (default: 3, 0=off)")
    parser.add_argument("--bg-voxel-size", type=float, default=0, help="Voxel downsample background points (0=off, e.g. 0.01)")
    parser.add_argument("--no-ply", action="store_true", help="Skip saving PLY files, only render video")
    parser.add_argument("--save-render-glb", action="store_true", help="Save per-frame GLB matching each rendered video frame (progressive bg + fg)")
    parser.add_argument("--save-combined-glb", action="store_true", help="Save a single GLB with fully merged background + all foreground points")
    parser.add_argument("--sphere-radius", type=float, default=0.005, help="Radius of each point sphere in GLB (default: 0.005)")
    parser.add_argument("--save-frames", type=str, default=None, help="Save individual fg/bg GLBs for specified frame range (e.g. '65-130')")
    parser.add_argument("--no-save-bg", action="store_true", help="Skip saving background GLB")
    args = parser.parse_args()

    # Validate input mode
    if args.bg_video and args.fg_video:
        if args.video:
            parser.error("Cannot use both positional video and --bg-video/--fg-video")
    elif not args.video:
        parser.error("Must provide either a single video or both --bg-video and --fg-video")

    # Parse --frames: detect paired format [fg,bg]--[fg,bg]--... vs normal
    paired_mode = False
    frame_indices = None
    bg_frame_indices = parse_frame_indices(args.bg_frames) if args.bg_frames else None

    sample_rate = args.sample_rate
    bg_sample_rate = args.bg_sample_rate

    if args.frames:
        pairs = parse_paired_frames(args.frames)
        if pairs is not None:
            # Paired mode: [fg,bg]--[fg,bg]--...
            paired_mode = True
            frame_indices = [p[0] for p in pairs]
            bg_frame_indices = [p[1] for p in pairs]
            sample_rate = 1
            bg_sample_rate = 1
            print(f"[paired mode] {len(pairs)} pairs: fg={frame_indices}, bg={bg_frame_indices}")
        else:
            frame_indices = parse_frame_indices(args.frames)

    # Support multiple fg videos and prompts (comma-separated)
    fg_videos = [v.strip() for v in args.fg_video.split(",")] if args.fg_video else [None]
    fg_prompts = [p.strip() for p in args.fg_prompt.split(",")]

    # Pad prompts/videos to match length
    if len(fg_videos) == 1 and len(fg_prompts) > 1:
        fg_videos = fg_videos * len(fg_prompts)
    if len(fg_prompts) == 1 and len(fg_videos) > 1:
        fg_prompts = fg_prompts * len(fg_videos)
    if len(fg_videos) != len(fg_prompts):
        parser.error(f"Number of fg videos ({len(fg_videos)}) must match number of fg prompts ({len(fg_prompts)})")

    save_frames_parsed = parse_frame_indices(args.save_frames) if args.save_frames else None

    for fg_vid, fg_pmt in zip(fg_videos, fg_prompts):
        out_dir = args.output_dir if len(fg_videos) == 1 else os.path.join(args.output_dir, fg_pmt)
        print(f"\n{'='*60}")
        print(f"Processing fg_video={fg_vid}, fg_prompt='{fg_pmt}', output={out_dir}")
        print(f"{'='*60}")
        run_stream3r_4d(
            video_path=args.video,
            bg_video_path=args.bg_video,
            fg_video_path=fg_vid,
            output_dir=out_dir,
            fg_prompt=fg_pmt,
            sample_rate=sample_rate,
            frame_indices=frame_indices,
            bg_sample_rate=bg_sample_rate,
            bg_frame_indices=bg_frame_indices,
            conf_percentile=args.conf_percentile,
            fg_conf_percentile=args.fg_conf_percentile,
            model_path=args.model_path,
            stream_mode=args.stream_mode,
            window_size=args.window_size,
            sam3_checkpoint=args.sam3_checkpoint,
            device=args.device,
            render_video=not args.no_render,
            render_fps=args.render_fps,
            cam_az=args.cam_az,
            cam_el=args.cam_el,
            cam_dist_scale=args.cam_dist,
            point_size=args.point_size,
            fov=args.fov,
            mask_dilate=args.mask_dilate,
            bg_voxel_size=args.bg_voxel_size,
            save_ply_files=not args.no_ply,
            save_render_glb=args.save_render_glb,
            save_combined_glb_flag=args.save_combined_glb,
            sphere_radius=args.sphere_radius,
            save_frames=save_frames_parsed,
            save_bg=not args.no_save_bg,
        )


if __name__ == "__main__":
    main()
