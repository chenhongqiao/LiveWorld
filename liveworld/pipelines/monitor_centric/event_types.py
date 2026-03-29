"""Event-centric data structures and helpers.

This module keeps event-related types minimal and explicit to avoid
implicit behaviors in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import hashlib

import numpy as np

# Event identifier type alias (string hash).
EventID = str


def _sha1_hex(payload: bytes) -> str:
    """Compute a stable SHA1 hex digest for ID generation."""
    return hashlib.sha1(payload).hexdigest()


def hash_pose(pose_c2w: np.ndarray) -> str:
    """Hash a 4x4 pose matrix for stable IDs.

    The pose is rounded to reduce floating-point jitter before hashing.
    """
    if pose_c2w.shape != (4, 4):
        raise ValueError(f"pose_c2w must be 4x4, got {pose_c2w.shape}")
    rounded = np.round(pose_c2w.astype(np.float64), 6)
    return _sha1_hex(rounded.tobytes())


def make_event_id(entities: List[str], anchor_pose_c2w: np.ndarray, seed: str = "") -> EventID:
    """Create a deterministic event ID based on entities and anchor pose."""
    if not entities:
        raise ValueError("entities must be non-empty to create event_id")
    # Keep entity order stable to avoid ID drift across runs.
    ent_key = ",".join([e.strip().lower() for e in entities])
    pose_key = hash_pose(anchor_pose_c2w)
    payload = f"{ent_key}|{pose_key}|{seed}".encode("utf-8")
    return _sha1_hex(payload)


@dataclass
class EntityDetectionResult:
    """Detection result for a single entity in a frame."""
    # Entity name (e.g., "car", "person").
    name: str
    # SAM3 segmentation mask for this entity (H, W), bool or uint8.
    mask: np.ndarray
    # Cropped entity image on black square background (H, W, 3), uint8.
    # Used for deduplication (DINOv3) and instance reference (observer conditioning).
    cropped_image: np.ndarray
    # Bounding box in original frame [x1, y1, x2, y2].
    bbox: Tuple[int, int, int, int]
    # Cropped SAM instance mask aligned with cropped_image (H, W), uint8/bool.
    # Used by deduplication to select foreground patches for embedding.
    cropped_mask: Optional[np.ndarray] = None


@dataclass
class EventObservation:
    """Observation of a potential event at a specific frame."""
    event_id: EventID
    frame_path: str
    frame_index: int
    pose_c2w: np.ndarray
    entities: List[str]
    timestamp: float
    # Frame as numpy array (H, W, 3), uint8 RGB. Used for detection.
    frame: Optional[np.ndarray] = None
    # Detection result with mask and cropped image.
    detection: Optional[EntityDetectionResult] = None
    # Optional full detection list for scene-level event registration.
    detections: Optional[List[EntityDetectionResult]] = None


@dataclass
class EventScript:
    """Script used to evolve an event in time."""
    event_id: EventID
    text: str
    horizon: int
    fps: int


@dataclass
class EventVideo:
    """Generated event video (fixed camera)."""
    event_id: EventID
    video_path: str
    fps: int
    num_frames: int
    anchor_pose_c2w: np.ndarray
    # Optional in-memory frames [T, H, W, 3] uint8.
    # When provided, downstream projection can avoid re-reading encoded MP4.
    frames: Optional[np.ndarray] = None


@dataclass
class EventPointCloud:
    """Event point cloud in the anchor camera coordinate system."""
    event_id: EventID
    points: np.ndarray
    colors: np.ndarray
    anchor_pose_c2w: np.ndarray
    frame_range: Tuple[int, int]
    # Per-frame foreground point clouds (list of N arrays, one per event frame).
    # Each entry is (M_t, 3) points for frame t. Used for temporally-aligned
    # fg projection where frame t shows the entity at its position at time t.
    per_frame_points: Optional[List[np.ndarray]] = None
    per_frame_colors: Optional[List[np.ndarray]] = None
    # Per-instance per-frame point clouds: {obj_id: [pts_t0, pts_t1, ...]}.
    per_instance_per_frame_points: Optional[Dict[int, List[np.ndarray]]] = None
    per_instance_per_frame_colors: Optional[Dict[int, List[np.ndarray]]] = None
    # Per-instance anchor masks (frame 0): {obj_id: (H,W) bool}.
    per_instance_anchor_masks: Optional[Dict[int, np.ndarray]] = None
    # Anchor frame (frame 0 of the I2V video) that per_instance_anchor_masks
    # were computed on.  Used by _get_instance_reference_frames so the mask
    # always matches the image regardless of agent state updates.
    anchor_frame: Optional[np.ndarray] = None  # (H, W, 3) uint8


@dataclass
class EventState:
    """Persistent state for a single event agent."""
    status: Literal["active", "ended", "paused"]
    anchor_pose_c2w: np.ndarray
    entities: List[str]
    # Current anchor frame for I2V generation.
    # For first iteration: the full scene frame (original input image).
    # For subsequent iterations: last frame of previous event video.
    current_anchor_frame: Optional[np.ndarray] = None
    # Path to current anchor frame (if saved to disk).
    current_anchor_frame_path: Optional[str] = None
    # Last generated video path.
    last_video_path: Optional[str] = None
    # Number of evolution iterations completed.
    iteration_count: int = 0
    # DINOv3 embedding for entity deduplication (torch.Tensor stored as Any to avoid import).
    entity_embedding: Optional[object] = None
    # Scene-level entity embeddings used for scene matching.
    scene_entity_embeddings: Optional[List[object]] = None
    # Cropped RGB instances aligned with scene_entity_embeddings.
    scene_entity_crops: Optional[List[np.ndarray]] = None
    # Cropped SAM masks aligned with scene_entity_embeddings.
    scene_entity_masks: Optional[List[np.ndarray]] = None
    # Entity labels aligned with scene_entity_embeddings.
    scene_entity_labels: Optional[List[str]] = None
    # Cached event target scene projection latent (fixed for this event camera).
    cached_target_scene_proj: Optional[object] = None
    # Cached decoded RGB frames from cached_target_scene_proj.
    cached_scene_proj_pixels: Optional[List[np.ndarray]] = None
    # Accumulated scene projection pixel frames across event iterations.
    # List of [H, W, 3] uint8 numpy arrays, decoded from target_scene_proj each iter.
    all_scene_proj_frames: Optional[List[np.ndarray]] = None
    # Accumulated fg projection pixel frames across event iterations.
    # List of [H, W, 3] uint8 numpy arrays, decoded from target_fg_proj each iter.
    all_fg_proj_frames: Optional[List[np.ndarray]] = None
    # All generated event frames across iterations (for preceding frame selection).
    # List of [H, W, 3] uint8 numpy arrays.
    all_generated_frames: Optional[List[np.ndarray]] = None
    # Permanent scene reference frames (fg masked out): set once after first iteration.
    # [7, H, W, 3] uint8 numpy array.
    reference_frames_scene: Optional[np.ndarray] = None
    # Permanent per-instance reference frames (fg crop on black bg): set once after first iteration.
    # [N_inst, H, W, 3] uint8 numpy array, one frame per detected instance.
    reference_frames_instance: Optional[np.ndarray] = None
    # Whether this event was created via intermediate detection (mid-round).
    is_intermediate: bool = False
    # Local frame index within the originating round where entity first appeared.
    intermediate_anchor_local_frame: Optional[int] = None
