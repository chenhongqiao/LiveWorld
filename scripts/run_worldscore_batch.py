"""LiveWorld worldscore baseline driver — JSON-driven, single-shot model load.

Mirrors `scripts/run_vbench_batch.py` but for the wm-agent worldscore layout
(`<root>/<name>/<split>/<traj>/{pose_string.txt, pose.json}`) and emits the
canonical `baseline_hywp` output structure (`<out>/<name>/<split>/<traj>/gen.mp4`).

Pipeline per (name, traj):
  1. Read `<worldscore_root>/<name>/<split>/<traj>/pose_string.txt` and convert
     the HYWP-style action string to per-frame `poses_c2w` via
     `generate_camera_trajectory_local` (same building block lyra2 / lingbot
     use). The `--forward-speed` knob (m/latent, default 0.08 = HYWP)
     scales translational actions; rotation is untouched.
  2. Write `<work>/<name>/geometry/<traj>.npz` containing only `poses_c2w`
     (LiveWorld's geometry loader supports this).
  3. Write `<work>/<name>/infer_scripts/<traj>.yaml` referring to that
     geometry, the source image, and the scene prompt.

Then `scripts/infer.py --configs-list <list>` is invoked once — model loads a
single time and sweeps every yaml. Finally each
`<output_root>/<name>/<traj>/final_video.mp4` is renamed to the canonical
`<final_out>/<name>/<split>/<traj>/gen.mp4`.

The original `<worldscore_root>` and the wm-agent dataset JSONs are not
modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Camera pose generation (mirrors lyra/Lyra-2/convert_action_to_lyra2.py)
# --------------------------------------------------------------------------- #

# Per-frame motion magnitudes. forward_speed is exposed as a CLI knob in
# m/latent units (so the canonical HYWP value is 0.08); we divide by
# FRAMES_PER_LATENT to get per-frame increments. Yaw/pitch are constant.
HYWP_FORWARD_SPEED = 0.08
YAW_SPEED = np.deg2rad(0.75)
PITCH_SPEED = np.deg2rad(0.75)
FRAMES_PER_LATENT = 4


def _load_hyworld_traj_fn():
    """Lazy-import generate_camera_trajectory_local from a configurable path."""
    hyvideo_dir = os.environ.get(
        "HYWORLDPLAY_HYVIDEO_DIR",
        "/data/harry/dynamic-wm/DynaTokenWM/HYWP/hyvideo",
    )
    if not os.path.isdir(hyvideo_dir):
        raise RuntimeError(
            f"hyvideo dir not found: {hyvideo_dir}. "
            f"Set HYWORLDPLAY_HYVIDEO_DIR to the path containing "
            f"generate_custom_trajectory.py.")
    if hyvideo_dir not in sys.path:
        sys.path.insert(0, hyvideo_dir)
    from generate_custom_trajectory import generate_camera_trajectory_local
    return generate_camera_trajectory_local


def parse_action_string(action_string: str, forward_speed: float):
    """Parse `<action>-<num_latents>[, ...]` → per-frame motion list."""
    fs = float(forward_speed) / FRAMES_PER_LATENT  # m/frame
    action_map = {
        "w":         {"forward":  fs},
        "s":         {"forward": -fs},
        "a":         {"right":   -fs},
        "d":         {"right":    fs},
        "up":        {"pitch":    PITCH_SPEED},
        "down":      {"pitch":   -PITCH_SPEED},
        "left":      {"yaw":     -YAW_SPEED},
        "right":     {"yaw":      YAW_SPEED},
        "rightup":   {"yaw":  YAW_SPEED, "pitch":  PITCH_SPEED},
        "rightdown": {"yaw":  YAW_SPEED, "pitch": -PITCH_SPEED},
        "leftup":    {"yaw": -YAW_SPEED, "pitch":  PITCH_SPEED},
        "leftdown":  {"yaw": -YAW_SPEED, "pitch": -PITCH_SPEED},
        "wd":        {"forward":  fs, "right":  fs},
        "dw":        {"forward":  fs, "right":  fs},
        "wa":        {"forward":  fs, "right": -fs},
        "aw":        {"forward":  fs, "right": -fs},
        "sd":        {"forward": -fs, "right":  fs},
        "ds":        {"forward": -fs, "right":  fs},
        "sa":        {"forward": -fs, "right": -fs},
        "as":        {"forward": -fs, "right": -fs},
        "wright":    {"forward":  fs, "yaw":   YAW_SPEED},
        "wleft":     {"forward":  fs, "yaw":  -YAW_SPEED},
        "sright":    {"forward": -fs, "yaw":   YAW_SPEED},
        "sleft":     {"forward": -fs, "yaw":  -YAW_SPEED},
        "dright":    {"right":    fs, "yaw":   YAW_SPEED},
        "dleft":     {"right":    fs, "yaw":  -YAW_SPEED},
        "aright":    {"right":   -fs, "yaw":   YAW_SPEED},
        "aleft":     {"right":   -fs, "yaw":  -YAW_SPEED},
        "wup":       {"forward":  fs, "pitch":  PITCH_SPEED},
        "wdown":     {"forward":  fs, "pitch": -PITCH_SPEED},
        "sup":       {"forward": -fs, "pitch":  PITCH_SPEED},
        "sdown":     {"forward": -fs, "pitch": -PITCH_SPEED},
    }
    motions, total_latents = [], 0
    for cmd in action_string.split(","):
        cmd = cmd.strip()
        if not cmd:
            continue
        action, n_lat = cmd.rsplit("-", 1)
        action = action.strip()
        n_lat = int(n_lat.strip())
        if action not in action_map:
            raise ValueError(f"Unknown action '{action}' in '{action_string}'")
        total_latents += n_lat
        for _ in range(n_lat * FRAMES_PER_LATENT):
            motions.append(action_map[action].copy())
    return motions, total_latents


def action_string_to_poses_c2w(action_string: str, forward_speed: float,
                                target_frames: int) -> np.ndarray:
    """Generate (target_frames, 4, 4) c2w camera poses from a HYWP action string."""
    gen_fn = _load_hyworld_traj_fn()
    motions, _ = parse_action_string(action_string, forward_speed)
    poses_c2w = np.array(gen_fn(motions), dtype=np.float32)
    # Pad/trim to target_frames; the generator returns len(motions)+1 poses.
    if len(poses_c2w) > target_frames:
        poses_c2w = poses_c2w[:target_frames]
    elif len(poses_c2w) < target_frames:
        pad = np.tile(poses_c2w[-1:], (target_frames - len(poses_c2w), 1, 1))
        poses_c2w = np.concatenate([poses_c2w, pad], axis=0)
    return poses_c2w


# --------------------------------------------------------------------------- #
# YAML helper (matches LiveWorld's expected schema)
# --------------------------------------------------------------------------- #

def build_case_yaml(yaml_path: Path, image_path: Path, geometry_path: Path,
                    scene_text: str, frames_per_iter: int) -> None:
    cfg = {
        "geometry_file_name": str(geometry_path.resolve()),
        "first_frame_image": str(image_path.resolve()),
        "observer": {"frames_per_iter": frames_per_iter},
        "iter_input": {
            "0": {"scene_text": scene_text.strip(), "fg_text": ""},
        },
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True,
                    help="Path to a worldscore-style JSON list of "
                         "{name, image, prompt, output} entries (only `name` is read).")
    ap.add_argument("--worldscore-root", required=True,
                    help="Per-instance dir root; expects "
                         "<root>/<name>/{init_16x9.png, video_prompt.txt, "
                         "<split>/<traj>/pose_string.txt}.")
    ap.add_argument("--output", required=True,
                    help="Final mp4 root; outputs land at "
                         "<output>/<name>/<split>/<traj>/gen.mp4.")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--names", default="",
                    help="Space-separated instance name filter (default: all).")
    ap.add_argument("--trajs", default="",
                    help="Space-separated trajectory name filter.")
    ap.add_argument("--forward-speed", "--forward_speed", type=float,
                    dest="forward_speed", default=HYWP_FORWARD_SPEED,
                    help=f"Translational speed in m/latent (default "
                         f"{HYWP_FORWARD_SPEED} = HYWP). Halve to 0.04 for slow "
                         f"cameras; yaw/pitch unaffected.")
    ap.add_argument("--frames-per-iter", type=int, default=61,
                    help="Frames per LiveWorld iteration. 15 latents → 61. "
                         "Override for different pose-string lengths.")
    ap.add_argument("--system-config", default="configs/infer_system_config_few_step_14B.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Generate cases + configs list; do not invoke infer.py.")
    args = ap.parse_args()

    # Resolve paths
    instances_json = Path(args.instances)
    ws_root = Path(args.worldscore_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / "_lw_work"
    lw_results_root = output_root / "_lw_results"
    work_root.mkdir(parents=True, exist_ok=True)
    lw_results_root.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]

    with open(instances_json) as f:
        instances = json.load(f)
    names_filter = set(args.names.split()) if args.names else None
    trajs_filter = set(args.trajs.split()) if args.trajs else None

    # Per-job final-output renaming bookkeeping. We stage scene_text into the
    # yaml's iter_input; LiveWorld's infer.py derives its own combo_id from
    # the yaml file stem, so we make the stem match the trajectory name.
    config_paths: List[str] = []
    rename_plan: List[Tuple[Path, Path, str, str, str]] = []  # (src, dst, name, split, traj)

    for inst in instances:
        name = inst.get("name") or inst.get("output")
        if not name:
            continue
        if names_filter is not None and name not in names_filter:
            continue
        inst_dir = ws_root / name
        img_src = inst_dir / "init_16x9.png"
        prompt_path = inst_dir / "video_prompt.txt"
        split_dir = inst_dir / args.split
        if not (img_src.exists() and prompt_path.exists() and split_dir.is_dir()):
            print(f"[skip] {name}: missing init/prompt/{args.split}", file=sys.stderr)
            continue
        scene_text = prompt_path.read_text().strip()

        scene_work = work_root / name
        scene_work.mkdir(parents=True, exist_ok=True)
        # Symlink source image (avoid copy; original is not touched).
        img_dst = scene_work / "source_image.png"
        if img_dst.is_symlink() or img_dst.exists():
            img_dst.unlink()
        img_dst.symlink_to(img_src.resolve())

        for traj_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            traj = traj_dir.name
            if trajs_filter is not None and traj not in trajs_filter:
                continue
            ps_path = traj_dir / "pose_string.txt"
            if not ps_path.exists():
                print(f"[skip] {name}/{traj}: no pose_string.txt", file=sys.stderr)
                continue

            canonical_dst = output_root / name / args.split / traj / "gen.mp4"
            if canonical_dst.exists():
                print(f"[skip] {name}/{traj}: gen.mp4 exists")
                continue

            action_string = ps_path.read_text().strip()
            poses_c2w = action_string_to_poses_c2w(
                action_string, args.forward_speed, args.frames_per_iter)
            geom_path = scene_work / "geometry" / f"{traj}.npz"
            geom_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(geom_path, poses_c2w=poses_c2w)

            yaml_path = scene_work / "infer_scripts" / f"{traj}.yaml"
            build_case_yaml(yaml_path=yaml_path, image_path=img_dst,
                            geometry_path=geom_path, scene_text=scene_text,
                            frames_per_iter=args.frames_per_iter)
            config_paths.append(str(yaml_path.resolve()))

            # LiveWorld's infer.py routes outputs to
            #   <output_root>/<image_stem>/<combo_id>/
            # where image_stem = name (parent-of-parent of the yaml) and
            # combo_id = yaml-file stem = traj.
            lw_combo_dir = lw_results_root / name / traj
            rename_plan.append((lw_combo_dir / "final_video.mp4",
                                canonical_dst, name, args.split, traj))

    print(f"[jobs] {len(config_paths)} (name, traj) tuples; "
          f"forward_speed={args.forward_speed} m/latent, "
          f"frames_per_iter={args.frames_per_iter}")

    list_path = work_root / "configs_list.txt"
    list_path.write_text("\n".join(config_paths) + "\n")
    if args.prepare_only or not config_paths:
        return

    # infer.py mutates --configs-list (deletes it after read); use a copy.
    runtime_list = work_root / "configs_list_runtime.txt"
    shutil.copy(list_path, runtime_list)

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "infer.py"),
        "--configs-list", str(runtime_list),
        "--system-config", args.system_config,
        "--output-root", str(lw_results_root),
        "--device", args.device,
    ]
    print("Running:", " ".join(cmd))
    # Don't check=True: LiveWorld occasionally segfaults during the
    # post-sampling finalize/save pass (Stream3R or stale C++ libs), but the
    # per-round observer/video.mp4 has already been written by then. We try
    # the canonical final_video.mp4 first and fall back to it.
    rc = subprocess.run(cmd, cwd=str(project_root)).returncode
    if rc != 0:
        print(f"[warn] infer.py exited rc={rc}; checking for round_0 fallbacks…",
              file=sys.stderr)

    # Promote final_video.mp4 (preferred) or round_0/observer/video.mp4 to
    # canonical baseline_hywp layout. With a single round (frames_per_iter
    # covers the whole trajectory), they have identical frame content.
    n_renamed = 0
    for src_final, dst, name, split, traj in rename_plan:
        # src_final = <lw_results_root>/<name>/<traj>/final_video.mp4
        # round-0 observer video is at <combo_dir>/round_0/observer/video.mp4
        round0_video = src_final.parent / "round_0" / "observer" / "video.mp4"
        if src_final.exists():
            picked = src_final
        elif round0_video.exists():
            picked = round0_video
        else:
            print(f"[warn] no mp4 for {name}/{split}/{traj}: tried "
                  f"{src_final} and {round0_video}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(picked, dst)
        n_renamed += 1
    print(f"[done] renamed {n_renamed}/{len(rename_plan)} mp4s into baseline_hywp layout.")


if __name__ == "__main__":
    main()
