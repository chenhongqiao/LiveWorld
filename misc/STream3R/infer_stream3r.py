"""Run STream3R on frames from one or more videos/images, save merged point cloud.

Usage:
    python misc/STream3R/infer_stream3r.py video1.mp4 video2.mp4 ...
    python misc/STream3R/infer_stream3r.py image.png
    python misc/STream3R/infer_stream3r.py video1.mp4 image.jpg --sample-rate 4 --output output.ply
    python misc/STream3R/infer_stream3r.py video1.mp4 --no-shuffle
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import open3d as o3d
import torch

from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.load_fn import load_and_preprocess_images


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def parse_frame_indices(spec: str) -> list[int]:
    """Parse a frame index specification like '0,1,2,3-10,15' into a sorted list of indices.

    Supports individual indices and inclusive ranges (e.g. 3-10 means 3,4,...,10).
    """
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


def is_image_file(path: str) -> bool:
    """Check if a file is an image based on extension."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def load_image(image_path: str) -> list[np.ndarray]:
    """Load a single image file as a list containing one RGB frame."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  {image_path}: 1 frame (single image)")
    return [frame]


def extract_frames_from_video(
    video_path: str,
    sample_rate: int = 1,
    frame_indices: list[int] | None = None,
) -> list[np.ndarray]:
    """Extract RGB frames from a video file.

    Args:
        video_path: Path to video file.
        sample_rate: Keep every N-th frame (ignored when frame_indices is set).
        frame_indices: If provided, extract only these specific frame indices.

    Returns:
        List of (H, W, 3) uint8 RGB frames.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_indices is not None:
        # Apply sample_rate on top of the selected frames
        sampled_indices = frame_indices[::sample_rate]
        wanted = set(sampled_indices)
        max_wanted = max(wanted) if wanted else -1
        frames = []
        idx = 0
        while cap.isOpened() and idx <= max_wanted:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        print(f"  {video_path}: {len(frames)} frames (selected {len(sampled_indices)} from {len(frame_indices)} specified, sample_rate={sample_rate}, {total} total)")
    else:
        frames = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        print(f"  {video_path}: {len(frames)} frames (sampled every {sample_rate} from {idx} total)")
    return frames


def frames_to_preprocessed_tensor(frames: list[np.ndarray], device: str = "cuda") -> torch.Tensor:
    """Convert list of RGB numpy frames to preprocessed Stream3R input tensor.

    Saves frames to temp files and uses load_and_preprocess_images, which
    handles the resizing/normalization that Stream3R expects.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for i, frame in enumerate(frames):
            path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            paths.append(path)
        images = load_and_preprocess_images(paths).to(device)
    return images


def run_stream3r(
    input_paths: list[str],
    output_path: str = "output_glbs/merged.ply",
    sample_rate: int = 1,
    frame_indices: list[int] | None = None,
    shuffle: bool = True,
    conf_percentile: float = 30.0,
    model_path: str = "hf_cache/yslan--STream3R",
    device: str = "cuda",
    seed: int = 42,
):
    """Run Stream3R on frames from videos/images and save point cloud.

    Args:
        input_paths: List of video or image file paths.
        output_path: Output PLY file path.
        sample_rate: Keep every N-th frame from each video (ignored for images).
        frame_indices: If provided, extract only these frame indices from videos.
        shuffle: Whether to shuffle frames across inputs.
        conf_percentile: Confidence percentile threshold (keep top N%).
        model_path: Path to STream3R pretrained model.
        device: Compute device.
        seed: Random seed for shuffle reproducibility.
    """
    # ---- 1. Extract frames from all inputs ----
    print(f"=== Extracting frames from {len(input_paths)} input(s) ===")
    all_frames = []
    frame_sources = []  # track which input each frame came from
    for path in input_paths:
        if is_image_file(path):
            frames = load_image(path)
        else:
            frames = extract_frames_from_video(path, sample_rate=sample_rate, frame_indices=frame_indices)
        for f in frames:
            all_frames.append(f)
            frame_sources.append(os.path.basename(path))

    print(f"Total frames: {len(all_frames)}")

    if len(all_frames) == 0:
        print("No frames extracted. Exiting.")
        return

    # ---- 2. Shuffle ----
    if shuffle:
        random.seed(seed)
        indices = list(range(len(all_frames)))
        random.shuffle(indices)
        all_frames = [all_frames[i] for i in indices]
        frame_sources = [frame_sources[i] for i in indices]
        print(f"Shuffled {len(all_frames)} frames (seed={seed})")

    # ---- 3. Preprocess and load model ----
    print("\n=== Preprocessing frames ===")
    images = frames_to_preprocessed_tensor(all_frames, device=device)
    print(f"Input tensor: {images.shape}")

    print(f"\n=== Loading STream3R model ===")
    model = STream3R.from_pretrained(model_path).to(device)
    model.eval()

    # ---- 4. Stream inference ----
    print(f"\n=== Running Stream3R ({len(all_frames)} frames) ===")
    session = StreamSession(model, mode="causal")

    collected_points = []
    collected_colors = []

    with torch.no_grad():
        for i in range(images.shape[0]):
            image = images[i : i + 1]
            predictions = session.forward_stream(image)

            preds = {}
            for key, val in predictions.items():
                if isinstance(val, torch.Tensor):
                    preds[key] = val.cpu().numpy().squeeze(0)
                else:
                    preds[key] = val

            # Extract current frame predictions (last in the accumulated sequence)
            wp = preds["world_points"]  # (N, H, W, 3)
            if wp.ndim == 4:
                wp = wp[-1:]  # take last frame
            conf = preds.get("world_points_conf", np.ones(wp.shape[:-1]))
            if conf.ndim >= 2 and conf.shape[0] == i + 1:
                conf = conf[-1:]
            img = preds["images"]
            if img.ndim == 4 and img.shape[0] == i + 1:
                img = img[-1:]
            if img.ndim == 4 and img.shape[1] == 3:
                img = np.transpose(img, (0, 2, 3, 1))

            pts = wp[0].reshape(-1, 3)
            cols = img[0].reshape(-1, 3)
            c = conf[0].reshape(-1) if conf.ndim >= 2 else conf.reshape(-1)

            # Confidence filtering
            thresh = max(np.percentile(c, conf_percentile), 1e-5)
            mask = c >= thresh
            collected_points.append(pts[mask])
            collected_colors.append(cols[mask])

            src = frame_sources[i] if i < len(frame_sources) else "?"
            print(f"  Frame {i}/{len(all_frames)-1} [{src}]: {mask.sum()} pts (top {100-conf_percentile:.0f}%)")

    session.clear()
    del model, session
    torch.cuda.empty_cache()

    # ---- 5. Merge and save ----
    print(f"\n=== Merging point cloud ===")
    merged_pts = np.concatenate(collected_points, axis=0)
    merged_cols = np.concatenate(collected_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    cols_norm = merged_cols.copy()
    if cols_norm.max() > 1.5:
        cols_norm = cols_norm / 255.0
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols_norm, 0, 1))

    print(f"Total points: {len(pcd.points):,}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run STream3R on video frames and save point cloud")
    parser.add_argument("inputs", nargs="+", help="One or more video/image paths")
    parser.add_argument("--sample-rate", type=int, default=1, help="Keep every N-th frame (default: 1)")
    parser.add_argument("--frames", type=str, default=None,
                        help="Select specific frames, e.g. '0,1,2,3-10,15'. Overrides --sample-rate.")
    parser.add_argument("--output", "-o", type=str, default="output_glbs/merged.ply", help="Output PLY path")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle frames across videos")
    parser.add_argument("--conf-percentile", type=float, default=30.0,
                        help="Confidence percentile threshold (default: 30, keeps top 70%%)")
    parser.add_argument("--model-path", type=str, default="hf_cache/yslan--STream3R")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    frame_indices = parse_frame_indices(args.frames) if args.frames else None

    run_stream3r(
        input_paths=args.inputs,
        output_path=args.output,
        sample_rate=args.sample_rate,
        frame_indices=frame_indices,
        shuffle=not args.no_shuffle,
        conf_percentile=args.conf_percentile,
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
