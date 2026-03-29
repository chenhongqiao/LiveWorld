"""
LiveWorld Dataset for Camera-Controlled Video Generation

LMDB结构 (每个sample):
- train_target_rgb.pt: [T, C, H, W] 目标视频latent
- train_target_scene_proj_rgb.pt: [T, C, H, W] 目标场景投影
- train_target_proj_fg_rgb.pt: [T, C, H, W] 目标前景投影 (可选)
- train_preceding_rgb_1.pt: [1, C, H, W] P1 preceding latent
- train_preceding_rgb_9.pt: [9, C, H, W] P9 preceding latent
- train_preceding_scene_proj_rgb_1.pt: [1, C, H, W] P1 preceding场景投影
- train_preceding_scene_proj_rgb_9.pt: [9, C, H, W] P9 preceding场景投影
- train_preceding_proj_fg_rgb_1.pt: [1, C, H, W] P1 preceding前景投影 (可选)
- train_preceding_proj_fg_rgb_9.pt: [9, C, H, W] P9 preceding前景投影 (可选)
- train_reference_rgb.pt: [R, C, H, W] reference frames (legacy [1, C*R, H, W] also supported)
- train_reference_scene_rgb.pt: [R, C, H, W] reference scene frames (optional, legacy supported)
- train_reference_instance_00.pt: [R, C, H, W] reference instance frames (optional, legacy supported)
- train_reference_instance_01.pt: [R, C, H, W] reference instance frames (optional, legacy supported)
- train_reference_instance_02.pt: [R, C, H, W] reference instance frames (optional, legacy supported)
- train_reference_instance_03.pt: [R, C, H, W] reference instance frames (optional, legacy supported)
- train_reference_instance_04.pt: [R, C, H, W] reference instance frames (optional, legacy supported)
- train_target_rgb.txt: 文本prompt
- train_target_scene_rgb.txt: 场景文本prompt (可选)
- train_target_fg_rgb.txt: 前景文本prompt (可选)
- train_target_rgb_frame0.png: 第一帧图像 (I2V)
- train_sample.json: metadata

P1/P9模式选择在process_batch中进行，dataset同时load两种数据。
"""

import os
import io
import pickle
import json
import torch
import lmdb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF


def _open_lmdb_env(path, readahead=False):
    return lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=readahead,
        meminit=False,
        max_readers=2048,
    )


class LiveWorldLMDBDataset(Dataset):
    """LiveWorld LMDB Dataset - loads both P1 and P9 data, mode selection in process_batch."""

    # IMPORTANT: Keep dataset logic minimal. Always load full data; no sampling/padding/truncation here.
    MAX_REF_INSTANCES = 5  # Max instance reference videos per sample

    def __init__(self, data_path: str, config=None, max_samples: int = None):
        self.data_path = data_path
        self.config = config
        self.max_samples = max_samples
        self.latent_channels = 16

        self.envs = None
        self.index = []

        # Try to load from cache first
        cache_file = os.path.join(data_path, "sharded_keys_cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            self.shard_paths = cache["shard_paths"]
            for shard_id, keys in enumerate(cache["shard_keys"]):
                for key in keys:
                    self.index.append((shard_id, key))
            print(f"[LiveWorldLMDBDataset] Loaded {cache['total_keys']} keys from cache")
        else:
            # Fallback: scan shards
            self.shard_paths = []
            for fname in sorted(os.listdir(data_path)):
                path = os.path.join(data_path, fname)
                if os.path.isdir(path) and fname.endswith('.lmdb'):
                    self.shard_paths.append(path)

            if not self.shard_paths:
                raise ValueError(f"No LMDB shards found in {data_path}")

            for shard_id, path in enumerate(self.shard_paths):
                tmp_env = _open_lmdb_env(path, readahead=False)
                with tmp_env.begin() as txn:
                    for key, _ in txn.cursor():
                        self.index.append((shard_id, key))
                tmp_env.close()
            print(f"[LiveWorldLMDBDataset] Scanned {len(self.index)} keys (no cache found)")

        if max_samples is not None and max_samples < len(self.index):
            self.index = self.index[:max_samples]

        print(f"[LiveWorldLMDBDataset] {len(self.index)} samples from {len(self.shard_paths)} shards")

    def __len__(self):
        return len(self.index)

    def reopen_envs(self):
        if self.envs is not None:
            return
        self.envs = [_open_lmdb_env(path, readahead=False) for path in self.shard_paths]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["envs"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.envs = None

    def _load_tensor(self, data: bytes) -> torch.Tensor:
        """Load tensor from bytes, handling dict format."""
        # Try weights_only=True first (faster), fallback to False if needed
        buffer = io.BytesIO(data)
        try:
            tensor_data = torch.load(buffer, weights_only=True)
        except Exception:
            buffer.seek(0)
            tensor_data = torch.load(buffer, weights_only=False)
        if isinstance(tensor_data, dict):
            return tensor_data['latent']
        return tensor_data

    def __getitem__(self, idx):
        self.reopen_envs()
        shard_id, key = self.index[idx]

        with self.envs[shard_id].begin() as txn:
            raw_data = txn.get(key)
            if raw_data is None:
                raise KeyError(f"Key {key} not found in shard {shard_id}")
            sample = pickle.loads(raw_data)

        # Target data
        target_latent = self._load_tensor(sample['train_target_rgb.pt'])
        target_scene_proj = self._load_tensor(sample['train_target_scene_proj_rgb.pt'])
        target_scene_proj_orig = self._load_tensor(sample['train_target_scene_proj_rgb_orig.pt']) if 'train_target_scene_proj_rgb_orig.pt' in sample else None
        target_proj_fg = self._load_tensor(sample['train_target_proj_fg_rgb.pt']) if 'train_target_proj_fg_rgb.pt' in sample else None

        # Load BOTH P1 and P9 preceding data
        preceding_latent_1 = self._load_tensor(sample['train_preceding_rgb_1.pt'])
        preceding_latent_9 = self._load_tensor(sample['train_preceding_rgb_9.pt'])
        preceding_scene_proj_1 = self._load_tensor(sample['train_preceding_scene_proj_rgb_1.pt'])
        preceding_scene_proj_9 = self._load_tensor(sample['train_preceding_scene_proj_rgb_9.pt'])

        # Optional foreground projections
        preceding_proj_fg_1 = self._load_tensor(sample['train_preceding_proj_fg_rgb_1.pt']) if 'train_preceding_proj_fg_rgb_1.pt' in sample else None
        preceding_proj_fg_9 = self._load_tensor(sample['train_preceding_proj_fg_rgb_9.pt']) if 'train_preceding_proj_fg_rgb_9.pt' in sample else None

        # Optional scene+fg overlay projections (pixel-level fg overlaid on scene)
        target_scene_proj_fg_overlay = self._load_tensor(sample['train_target_scene_proj_fg_overlay_rgb.pt']) if 'train_target_scene_proj_fg_overlay_rgb.pt' in sample else None
        preceding_scene_proj_fg_overlay_1 = self._load_tensor(sample['train_preceding_scene_proj_fg_overlay_rgb_1.pt']) if 'train_preceding_scene_proj_fg_overlay_rgb_1.pt' in sample else None
        preceding_scene_proj_fg_overlay_9 = self._load_tensor(sample['train_preceding_scene_proj_fg_overlay_rgb_9.pt']) if 'train_preceding_scene_proj_fg_overlay_rgb_9.pt' in sample else None

        # NOTE: Dataset must always load full reference data.
        # Do NOT pad/sample/truncate here; handle MAX_REF_SCENE_FRAMES in process_batch.
        H, W = target_latent.shape[-2:]
        reference_latent = torch.zeros(0, self.latent_channels, H, W, dtype=target_latent.dtype)
        if 'train_reference_rgb.pt' in sample:
            reference_latent = self._unpack_reference_frames(
                self._load_tensor(sample['train_reference_rgb.pt'])
            )
            if reference_latent is None or reference_latent.numel() == 0:
                reference_latent = torch.zeros(0, self.latent_channels, H, W, dtype=target_latent.dtype)

        # Optional reference scene latents
        if 'train_reference_scene_rgb.pt' in sample:
            reference_scene_latent = self._unpack_reference_frames(
                self._load_tensor(sample['train_reference_scene_rgb.pt'])
            )
            if reference_scene_latent is None or reference_scene_latent.numel() == 0:
                reference_scene_latent = torch.zeros(0, self.latent_channels, H, W, dtype=target_latent.dtype)
        else:
            # Fallback: treat reference_rgb as scene-only when explicit scene reference is missing.
            reference_scene_latent = reference_latent

        # Optional non-augmented reference scene latents
        if 'train_reference_scene_rgb_orig.pt' in sample:
            reference_scene_latent_orig = self._unpack_reference_frames(
                self._load_tensor(sample['train_reference_scene_rgb_orig.pt'])
            )
            if reference_scene_latent_orig is None or reference_scene_latent_orig.numel() == 0:
                reference_scene_latent_orig = None
        else:
            reference_scene_latent_orig = None

        # Optional reference instance latents (keep per-instance frames)
        reference_instance_latents = []
        for idx in range(self.MAX_REF_INSTANCES):
            key = f"train_reference_instance_{idx:02d}.pt"
            if key in sample:
                frames = self._unpack_reference_frames(self._load_tensor(sample[key]))
                if frames is not None and frames.numel() > 0:
                    reference_instance_latents.append(frames)

        # Text prompt
        prompt = sample['train_target_rgb.txt']
        if isinstance(prompt, bytes):
            prompt = prompt.decode()

        scene_prompt = prompt
        if 'train_target_scene_rgb.txt' in sample:
            scene_prompt = sample['train_target_scene_rgb.txt']
            if isinstance(scene_prompt, bytes):
                scene_prompt = scene_prompt.decode()

        fg_prompt = ""
        if 'train_target_fg_rgb.txt' in sample:
            fg_prompt = sample['train_target_fg_rgb.txt']
            if isinstance(fg_prompt, bytes):
                fg_prompt = fg_prompt.decode()

        # First frame image for I2V
        img = Image.open(io.BytesIO(sample['train_target_rgb_frame0.png'])).convert('RGB')
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        # Metadata
        meta = sample['train_sample.json']
        if isinstance(meta, bytes):
            meta = json.loads(meta)
        has_fg = bool(meta.get("has_fg", False))
        if "has_fg" not in meta:
            has_fg = bool(fg_prompt.strip()) or (target_proj_fg is not None)

        return {
            "idx": idx,
            "prompts": prompt,
            "scene_prompts": scene_prompt,
            "fg_prompts": fg_prompt,
            "has_fg": has_fg,
            # Target
            "target_latent": target_latent.float(),
            "target_scene_proj": target_scene_proj.float(),
            "target_scene_proj_orig": target_scene_proj_orig.float() if target_scene_proj_orig is not None else None,
            "target_proj_fg": target_proj_fg.float() if target_proj_fg is not None else None,
            "target_scene_proj_fg_overlay": target_scene_proj_fg_overlay.float() if target_scene_proj_fg_overlay is not None else None,
            # P1 preceding (1 frame)
            "preceding_latent_1": preceding_latent_1.float(),
            "preceding_scene_proj_1": preceding_scene_proj_1.float(),
            "preceding_proj_fg_1": preceding_proj_fg_1.float() if preceding_proj_fg_1 is not None else None,
            "preceding_scene_proj_fg_overlay_1": preceding_scene_proj_fg_overlay_1.float() if preceding_scene_proj_fg_overlay_1 is not None else None,
            # P9 preceding (9 frames)
            "preceding_latent_9": preceding_latent_9.float(),
            "preceding_scene_proj_9": preceding_scene_proj_9.float(),
            "preceding_proj_fg_9": preceding_proj_fg_9.float() if preceding_proj_fg_9 is not None else None,
            "preceding_scene_proj_fg_overlay_9": preceding_scene_proj_fg_overlay_9.float() if preceding_scene_proj_fg_overlay_9 is not None else None,
            # Reference (full load, no padding)
            "reference_latent": reference_latent.float(),
            "reference_scene_latent": reference_scene_latent.float(),
            "reference_scene_latent_orig": reference_scene_latent_orig.float() if reference_scene_latent_orig is not None else None,
            "reference_instance_latents": reference_instance_latents,
            # I2V
            "img": img,
            "meta": meta,
        }

    def _unpack_reference_frames(self, raw: torch.Tensor) -> torch.Tensor:
        """Normalize reference latent to [R, C, H, W]."""
        if raw is None:
            return torch.zeros(0, self.latent_channels, 0, 0)
        if raw.dim() == 3:
            return raw.unsqueeze(0)
        if raw.dim() == 5 and raw.shape[0] == 1:
            return raw.squeeze(0)
        if raw.dim() == 4:
            # [1, C*R, H, W]
            if raw.shape[0] == 1:
                _, total_channels, H, W = raw.shape
                num_frames = total_channels // self.latent_channels
                if num_frames <= 0:
                    return torch.zeros(0, self.latent_channels, H, W, dtype=raw.dtype)
                return raw.squeeze(0).view(num_frames, self.latent_channels, H, W)
            # [R, C, H, W]
            if raw.shape[1] == self.latent_channels:
                return raw
            # [C, R, H, W]
            if raw.shape[0] == self.latent_channels:
                return raw.permute(1, 0, 2, 3)
        return raw


def liveworld_collate_fn(batch):
    """Custom collate function - stacks fixed-size tensors, keeps variable-length references as lists."""

    def _collate_optional(key):
        items = [b.get(key) for b in batch]
        if all(t is None for t in items):
            return None
        if all(t is not None for t in items):
            return torch.stack(items)
        return items

    return {
        "idx": torch.tensor([b["idx"] for b in batch]),
        "prompts": [b["prompts"] for b in batch],
        "scene_prompts": [b["scene_prompts"] for b in batch],
        "fg_prompts": [b["fg_prompts"] for b in batch],
        "has_fg": torch.tensor([b["has_fg"] for b in batch], dtype=torch.bool),
        "meta": [b["meta"] for b in batch],
        # Target
        "target_latent": torch.stack([b["target_latent"] for b in batch]),
        "target_scene_proj": torch.stack([b["target_scene_proj"] for b in batch]),
        "target_scene_proj_orig": _collate_optional("target_scene_proj_orig"),
        "target_proj_fg": _collate_optional("target_proj_fg"),
        "target_scene_proj_fg_overlay": _collate_optional("target_scene_proj_fg_overlay"),
        # P1 preceding (fixed 1 frame)
        "preceding_latent_1": torch.stack([b["preceding_latent_1"] for b in batch]),
        "preceding_scene_proj_1": torch.stack([b["preceding_scene_proj_1"] for b in batch]),
        "preceding_proj_fg_1": _collate_optional("preceding_proj_fg_1"),
        "preceding_scene_proj_fg_overlay_1": _collate_optional("preceding_scene_proj_fg_overlay_1"),
        # P9 preceding (fixed 9 frames)
        "preceding_latent_9": torch.stack([b["preceding_latent_9"] for b in batch]),
        "preceding_scene_proj_9": torch.stack([b["preceding_scene_proj_9"] for b in batch]),
        "preceding_proj_fg_9": _collate_optional("preceding_proj_fg_9"),
        "preceding_scene_proj_fg_overlay_9": _collate_optional("preceding_scene_proj_fg_overlay_9"),
        # Reference (full load, handled in process_batch)
        "reference_latent": [b["reference_latent"] for b in batch],
        "reference_scene_latent": [b["reference_scene_latent"] for b in batch],
        "reference_scene_latent_orig": [b.get("reference_scene_latent_orig") for b in batch],
        "reference_instance_latents": [b["reference_instance_latents"] for b in batch],
        # I2V
        "img": torch.stack([b["img"] for b in batch]),
    }


if __name__ == "__main__":
    data_path = "data/liveworld/frame33_fps16_40000_lmdb"
    dataset = LiveWorldLMDBDataset(data_path)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=8,
        collate_fn=liveworld_collate_fn,
        prefetch_factor=2,
        persistent_workers=True,
    )

    for batch in tqdm(loader, desc="Testing dataloader"):
        pass

    print("Done! All samples loaded successfully.")
