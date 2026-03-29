"""
FMWan Task - Flow Matching Video Generation Task

This module defines the FMWan task which handles:
- Model initialization (generator, text_encoder, vae, image_encoder)
- Batch processing (process_batch)
- Loss computation (generator_fm_loss)
- Trainable parameter management (get_trainable_parameters)
- Validation inference (validate) - can be overridden by subclasses

Model Loading Flow (simplified):
    1. Wrapper.__init__ loads backbone:
       - Random initialization
       - Official Wan pretrained weights (wan_model_name)
       - Backbone checkpoint (generator_backbone_ckpt_path, only if use_lora=False)

    2. Task._load_submodule_weights() loads submodules (before LoRA):
       - control_adapter, sp_blocks, etc.
       - Hook for subclasses to override

    3. Task._apply_and_load_lora() applies LoRA (if use_lora=True):
       - Apply LoRA via get_peft_model
       - Load LoRA weights from generator_lora_ckpt_path (if provided)

    4. Task._initialize_encoders() loads encoders:
       - text_encoder, vae, image_encoder

NOTE: Backbone and LoRA checkpoints are now separate:
  - generator_backbone_ckpt_path: Full backbone weights (when use_lora=False)
  - generator_lora_ckpt_path: LoRA adapter weights (when use_lora=True)
Training loop logic (save, visualization, logging) is handled by the Trainer.
"""

import os
import random
import math
import glob
import json
import dataclasses
import traceback
import cv2
import imageio
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from PIL import Image
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from omegaconf import OmegaConf
from safetensors.torch import load_file

from .wrapper import BidirectionalWanWrapperSP, WanTextEncoder, WanVAEWrapper, WanCLIPEncoder
from .utils import get_denoising_loss
from .utils import prepare_for_saving
from .utils import (
    crop_latent_frames,
    process_timestep,
    get_timestep,
    diffusion_forcing as diffusion_forcing_util,
    get_full_state_dict,
    create_eval_generator,
)
from liveworld.geometry_utils import BackboneInferenceOptions
from liveworld.pipelines.pipeline_unified_backbone import (
    UnifiedBackbonePipeline,
    run_iterative_inference,
    load_frame_from_image,
)


class FMWanLiveWorld(nn.Module):
    """
    LiveWorld: Camera-controlled video generation with scene point cloud conditioning.

    Architecture (LiveWorld paper) with [T, P, R] frame order:
    - Generator input: [noisy_target_latent, preceding_latent, reference_latent] concat along frame dim
    - State Adapter input: [target_scene_proj, preceding_scene_proj] concat along frame dim (T+P frames)
    - State Adapter hints are added to FIRST T+P tokens (since T and P frames come first in [T, P, R] order)

    Frame order: [T, P, R] where:
    - T (target): noisy frames with sampled timestep - to be generated
    - P (preceding): clean frames with timestep=0 - temporal context
    - R (reference): clean frames with timestep=0 - appearance reference

    Training data structure:
    - target_latent: [B, T, C, H, W] - frames to generate
    - target_scene_proj: [B, T, C, H, W] - scene projection for target views
    - preceding_latent: [B, P, C, H, W] - preceding frames (synchronized dropout with scene proj)
    - preceding_scene_proj: [B, P, C, H, W] - scene projection for preceding views
    - reference_latent: [B, R, C, H, W] - reference frames (random dropout)
    - img: [B, 3, H, W] - first frame image for I2V
    """

    def __init__(self, config, device):
        super().__init__()

        # Training stage name (set by trainer via set_training_stage)
        self.current_stage_name = "state_adapter"
        self._lora_weights_loaded = False


        self.device = device
        self.config = config
        self.dtype = torch.bfloat16 if self.config.mixed_precision else torch.float32

        self._initialize_models(device)
        self._initialize_scheduler(device)

        if config.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.seq_len = config.image_or_video_shape[1] * config.image_or_video_shape[3] * config.image_or_video_shape[4] // (2*2)

        self.num_frame_per_block = config.num_frame_per_block
        self.backbone_unlocked = False

    def _initialize_scheduler(self, device):
        """Initialize scheduler and related hyperparameters."""
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.num_train_timestep = self.config.num_train_timestep
        self.timestep_shift = getattr(self.config, "timestep_shift", 1.0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        self.denoising_loss_func = get_denoising_loss(self.config.denoising_loss_type)()

        # By default, full timestep
        self.denoising_step_list = self.scheduler.timesteps

        # If few-step, override
        if hasattr(self.config, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(
                self.config.denoising_step_list, dtype=torch.long, device=self.device
            )
            if self.config.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    #! ==================== Trainable Parameters ====================

    def _get_eval_encoders(self):
        """Get or create evaluation encoders (non-FSDP) for validation."""
        if not hasattr(self, "_eval_text_encoder"):
            self._eval_text_encoder = WanTextEncoder(model_name=self.config.wan_model_name).to(
                device=self.device, dtype=self.dtype
            )
            self._eval_text_encoder.requires_grad_(False)
        
        clip_model = None
        if self.config.i2v:
            if not hasattr(self, "_eval_clip_encoder"):
                self._eval_clip_encoder = WanCLIPEncoder(model_name=self.config.wan_model_name).to(
                    device=self.device, dtype=self.dtype
                )
                self._eval_clip_encoder.requires_grad_(False)
            clip_model = self._eval_clip_encoder

        return self._eval_text_encoder, clip_model

    def set_training_stage(self, stage_name: str):
        """Called by trainer when switching training stages."""
        self.current_stage_name = stage_name
        print(f"[FMWanLiveWorld] Training stage set to: {stage_name}")
        if stage_name.lower() == "lora" and getattr(self.config, "use_lora", False):
            if not self._lora_weights_loaded:
                self._load_lora_weights()
                self._lora_weights_loaded = True

    def _initialize_models(self, device):
        """Initialize all models for LiveWorld training."""
        # Initialize generator with State Adapter support
        self.generator = BidirectionalWanWrapperSP(config=self.config, role="generator")

        # Load State Adapter weights if provided
        self._load_sp_weights()

        # Optionally register LoRA on main backbone only (excluding State Adapter)
        if getattr(self.config, "use_lora", False):
            lora_config_dict = dict(self.config.lora_config)

            # Collect only main backbone modules (exclude State Adapter)
            original_targets = lora_config_dict.get("target_modules", ["q", "k", "v", "o"])
            backbone_modules = []
            for name, _ in self.generator.named_modules():
                # Skip State Adapter modules
                if 'sp_' in name.lower():
                    continue
                # Check if module name ends with any target
                for target in original_targets:
                    if name.endswith(f".{target}"):
                        backbone_modules.append(name)
                        break

            if backbone_modules:
                lora_config_dict["target_modules"] = backbone_modules
                print(f"[LiveWorld] LoRA target modules: {len(backbone_modules)} backbone modules (State Adapter excluded)")
            else:
                print(f"[LiveWorld] WARNING: No backbone modules found for LoRA, using original targets")

            lora_config = LoraConfig(**lora_config_dict)
            self.generator = get_peft_model(self.generator, lora_config)
            for _, cfg in self.generator.peft_config.items():
                cfg.inference_mode = False

            # Load pretrained LoRA weights if provided
            stage_name = getattr(self, "current_stage_name", "state_adapter")
            if stage_name.lower() == "lora":
                self._load_lora_weights()
                self._lora_weights_loaded = True

        # Start with all parameters frozen; stages will toggle grads as needed
        self.generator.requires_grad_(False)

        # Initialize frozen encoders and decoder
        self.text_encoder = WanTextEncoder(model_name=self.config.wan_model_name)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=self.config.wan_model_name)
        self.vae.requires_grad_(False)

        if self.config.i2v:
            self.image_encoder = WanCLIPEncoder(model_name=self.config.wan_model_name)
            self.image_encoder.requires_grad_(False)

        # Setup scheduler
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _load_sp_weights(self):
        """
        Load pretrained State Adapter weights (sp_blocks and sp_patch_embedding).

        Supports:
        - Single .safetensors file
        - Single .pt/.pth file
        - Directory with sharded safetensors (like HuggingFace format)

        Only loads weights where the shape matches between checkpoint and model.
        This allows loading partial weights when architecture differs (e.g., sp_in_dim changed).
        """
        sp_model_path = getattr(self.config, "sp_model_path", None)
        if sp_model_path is None:
            print("[Task] No sp_model_path provided, State Adapter blocks remain randomly initialized")
            return

        if not os.path.exists(sp_model_path):
            print(f"[Task] sp_model_path '{sp_model_path}' does not exist, State Adapter blocks remain randomly initialized")
            return

        print(f"[Task] Loading State Adapter weights from: {sp_model_path}")

        # Load checkpoint - support directory, single file, or sharded safetensors
        checkpoint = {}

        if os.path.isdir(sp_model_path):
            # Directory: check for sharded safetensors or single file
            index_file = os.path.join(sp_model_path, "diffusion_pytorch_model.safetensors.index.json")
            single_safetensor = os.path.join(sp_model_path, "diffusion_pytorch_model.safetensors")

            if os.path.exists(index_file):
                # Sharded safetensors with index file
                print(f"[Task]   Found sharded safetensors with index file")
                with open(index_file, "r") as f:
                    index_data = json.load(f)

                # Get unique shard files
                weight_map = index_data.get("weight_map", {})
                shard_files = set(weight_map.values())

                # Only load State Adapter-related shards
                sp_keys_in_shards = {k: v for k, v in weight_map.items()
                                       if "sp_blocks" in k or "sp_patch_embedding" in k}

                if sp_keys_in_shards:
                    sp_shard_files = set(sp_keys_in_shards.values())
                    print(f"[Task]   Loading {len(sp_shard_files)} shard(s) containing State Adapter weights")
                    for shard_file in sp_shard_files:
                        shard_path = os.path.join(sp_model_path, shard_file)
                        if os.path.exists(shard_path):
                            shard_weights = load_file(shard_path, device="cpu")
                            checkpoint.update(shard_weights)
                else:
                    print(f"[Task]   No State Adapter keys found in index, loading all shards")
                    for shard_file in shard_files:
                        shard_path = os.path.join(sp_model_path, shard_file)
                        if os.path.exists(shard_path):
                            shard_weights = load_file(shard_path, device="cpu")
                            checkpoint.update(shard_weights)

            elif os.path.exists(single_safetensor):
                # Single safetensors file in directory
                checkpoint = load_file(single_safetensor, device="cpu")
            else:
                # Try to find any safetensors files
                safetensor_files = glob.glob(os.path.join(sp_model_path, "*.safetensors"))
                if safetensor_files:
                    print(f"[Task]   Found {len(safetensor_files)} safetensors file(s)")
                    for sf in safetensor_files:
                        shard_weights = load_file(sf, device="cpu")
                        checkpoint.update(shard_weights)
                else:
                    # Try .pt or .pth files
                    pt_files = glob.glob(os.path.join(sp_model_path, "*.pt")) + \
                               glob.glob(os.path.join(sp_model_path, "*.pth"))
                    if pt_files:
                        checkpoint = torch.load(pt_files[0], map_location="cpu", weights_only=False)
                    else:
                        print(f"[Task] No loadable weights found in {sp_model_path}")
                        return

        elif sp_model_path.endswith(".safetensors"):
            checkpoint = load_file(sp_model_path, device="cpu")
        else:
            checkpoint = torch.load(sp_model_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "generator" in checkpoint:
            state_dict = checkpoint["generator"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        elif "generator_ema" in checkpoint:
            state_dict = checkpoint["generator_ema"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        # Get current model state dict for shape comparison
        model_state_dict = self.generator.model.state_dict()

        # Filter State Adapter-related keys and check shape compatibility
        sp_keys = ["sp_blocks", "sp_patch_embedding"]
        sp_state_dict = {}
        skipped_keys = []

        def clean_sp_key(key: str) -> str:
            """Clean checkpoint key to match current model structure.

            Handles:
            - Old format with LoRA on State Adapter: base_model.model.model.sp_blocks.0.self_attn.q.base_layer.weight
            - New format without LoRA: model.sp_blocks.0.self_attn.q.weight
            - FSDP format: base_sp_blocks.0.xxx -> sp_blocks.0.xxx
            """
            # Skip LoRA-specific keys (lora_A, lora_B) - we only want base weights
            if 'lora_A' in key or 'lora_B' in key:
                return None

            # Remove common prefixes
            clean = key
            for prefix in ['base_model.model.', 'base_model.', 'model.']:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]

            # Handle LoRA base_layer -> original weight
            # e.g., sp_blocks.0.self_attn.q.base_layer.weight -> sp_blocks.0.self_attn.q.weight
            clean = clean.replace('.base_layer.', '.')

            # Remove any remaining 'model.' prefix
            if clean.startswith('model.'):
                clean = clean[6:]

            # Handle FSDP/checkpoint format: base_sp_blocks -> sp_blocks
            clean = clean.replace('base_sp_blocks', 'sp_blocks')
            clean = clean.replace('base_sp_patch_embedding', 'sp_patch_embedding')

            return clean

        for key, value in state_dict.items():
            clean_key = clean_sp_key(key)
            if clean_key is None:  # Skip LoRA-specific keys
                continue
            if any(sp_key in clean_key for sp_key in sp_keys):
                # Check if key exists in model and shapes match
                if clean_key in model_state_dict:
                    model_shape = model_state_dict[clean_key].shape
                    ckpt_shape = value.shape
                    if model_shape == ckpt_shape:
                        sp_state_dict[clean_key] = value
                    else:
                        skipped_keys.append(f"{clean_key}: ckpt {list(ckpt_shape)} vs model {list(model_shape)}")
                else:
                    skipped_keys.append(f"{clean_key}: not in model")

        if not sp_state_dict:
            print(f"[Task] WARNING: No compatible State Adapter weights found in {sp_model_path}")
            if skipped_keys:
                print(f"[Task]   Skipped (shape mismatch): {skipped_keys[:5]}...")
            return

        # Load matched State Adapter weights into model
        missing_keys, unexpected_keys = self.generator.model.load_state_dict(sp_state_dict, strict=False)

        # Filter missing_keys to only show State Adapter-related keys
        sp_missing_keys = [k for k in missing_keys if any(sp_key in k for sp_key in sp_keys)]

        print(f"[Task] ✓ Loaded {len(sp_state_dict)} State Adapter parameters (shape-matched)")
        if skipped_keys:
            print(f"[Task]   Skipped {len(skipped_keys)} params (shape mismatch):")
            for sk in skipped_keys[:10]:
                print(f"[Task]     - {sk}")
            if len(skipped_keys) > 10:
                print(f"[Task]     ... and {len(skipped_keys) - 10} more")
        if sp_missing_keys:
            print(f"[Task]   Missing State Adapter keys: {sp_missing_keys[:5]}...")

        del state_dict, sp_state_dict
        torch.cuda.empty_cache()

    def _load_lora_weights(self):
        """
        Load pretrained LoRA weights from a checkpoint file.

        Only loads LoRA parameters (keys containing 'lora_').
        State Adapter weights should be loaded separately via sp_model_path.
        """
        lora_model_path = getattr(self.config, "lora_model_path", None)
        if lora_model_path is None:
            return

        if not os.path.exists(lora_model_path):
            print(f"[Task] lora_model_path '{lora_model_path}' does not exist, LoRA remains randomly initialized")
            return

        print(f"[Task] Loading LoRA weights from: {lora_model_path}")

        # Load checkpoint
        if os.path.isdir(lora_model_path):
            # Directory: look for model.pt or model.safetensors
            candidates = ["model.pt", "model.safetensors", "checkpoint.pt"]
            checkpoint = None
            for candidate in candidates:
                candidate_path = os.path.join(lora_model_path, candidate)
                if os.path.exists(candidate_path):
                    if candidate.endswith(".safetensors"):
                        checkpoint = {"generator": load_file(candidate_path, device="cpu")}
                    else:
                        checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
                    print(f"[Task]   Loaded from: {candidate_path}")
                    break
            if checkpoint is None:
                print(f"[Task] No checkpoint file found in {lora_model_path}")
                return
        elif lora_model_path.endswith(".safetensors"):
            checkpoint = {"generator": load_file(lora_model_path, device="cpu")}
        else:
            checkpoint = torch.load(lora_model_path, map_location="cpu", weights_only=False)

        # Extract state dict
        if "generator" in checkpoint:
            state_dict = checkpoint["generator"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Extract only LoRA parameters
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora_" in k}

        # Load LoRA weights
        if lora_state_dict:
            set_peft_model_state_dict(self.generator, lora_state_dict)
            print(f"[Task] ✓ Loaded {len(lora_state_dict)} LoRA parameters")
        else:
            print(f"[Task] WARNING: No LoRA parameters found in checkpoint")

        del state_dict, lora_state_dict
        torch.cuda.empty_cache()

    def get_trainable_parameters(self, stage="state_adapter"):
        """Return trainable parameters based on stage."""
        # Freeze all parameters first
        self.generator.model.requires_grad_(False)
        trainable_params = []

        if stage == "state_adapter" or stage == "controlnet":
            # Only train State Adapter/ControlNet-related parameters
            for name, param in self.generator.model.named_parameters():
                if 'sp_' in name.lower():
                    param.requires_grad_(True)
                    trainable_params.append(param)
            print(f"[LiveWorld] Training stage '{stage}': State Adapter parameters only")

        elif stage == "backbone_only":
            # Train only backbone (excluding State Adapter modules)
            for name, param in self.generator.model.named_parameters():
                if 'sp_' not in name.lower():
                    param.requires_grad_(True)
                    trainable_params.append(param)
            print(f"[LiveWorld] Training stage '{stage}': Backbone parameters only")

        elif stage in ("all", "full"):
            # Train all parameters
            self.generator.model.requires_grad_(True)
            trainable_params.extend(self.generator.model.parameters())
            print(f"[LiveWorld] Training stage '{stage}': All parameters")

        elif stage == "lora":
            # Train only LoRA modules
            if hasattr(self.generator, "peft_config"):
                for name, param in self.generator.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
                        trainable_params.append(param)
                print(f"[LiveWorld] Training stage '{stage}': LoRA parameters only")
            else:
                raise ValueError("LoRA is not configured but stage is set to 'lora'")

        else:
            raise NotImplementedError(f"Training stage '{stage}' is not implemented")

        return trainable_params
    
    def _get_fg_mode_probs(self) -> dict:
        """
        Get stage-aware foreground mode probabilities.

        Config keys (optional):
        - fg_mode_probs_sp
        - fg_mode_probs_lora
        - fg_mode_probs_default

        Each key is a mapping with optional entries:
        {fg_text_only, fg_proj_only, both}
        """
        stage_name = str(getattr(self, "current_stage_name", "")).lower()
        mode_keys = ("fg_text_only", "fg_proj_only", "both")

        if stage_name in ("state_adapter", "controlnet"):
            base = {"fg_text_only": 0.0, "fg_proj_only": 0.6, "both": 0.4}
            cfg_probs = getattr(self.config, "fg_mode_probs_sp", None)
        elif stage_name == "lora":
            base = {"fg_text_only": 0.3, "fg_proj_only": 0.3, "both": 0.4}
            cfg_probs = getattr(self.config, "fg_mode_probs_lora", None)
        else:
            base = {"fg_text_only": 1.0 / 3.0, "fg_proj_only": 1.0 / 3.0, "both": 1.0 / 3.0}
            cfg_probs = getattr(self.config, "fg_mode_probs_default", None)

        probs = dict(base)
        if cfg_probs is not None:
            for k in mode_keys:
                if k in cfg_probs:
                    try:
                        probs[k] = float(cfg_probs[k])
                    except (TypeError, ValueError):
                        pass

        probs = {k: max(0.0, float(v)) for k, v in probs.items()}
        total = sum(probs.values())
        if total <= 0:
            return {"fg_text_only": 1.0 / 3.0, "fg_proj_only": 1.0 / 3.0, "both": 1.0 / 3.0}
        return {k: v / total for k, v in probs.items()}

    def _select_fg_mode(
        self,
        has_fg: bool,
        fg_text: str,
        fg_proj_available: bool,
    ) -> str:
        # Foreground mode selection priority:
        # 1) has_fg=False -> force no-foreground behavior (scene_only)
        # 2) both fg text and fg projection available -> sample by stage-aware probs
        # 3) only one condition available -> use the available one
        # 4) neither available -> scene_only
        fg_text_available = bool(fg_text.strip())
        fg_proj_ok = fg_proj_available
        if not has_fg:
            fg_text_available = False
            fg_proj_ok = False

        if fg_text_available and fg_proj_ok:
            probs = self._get_fg_mode_probs()
            mode_order = ["fg_text_only", "fg_proj_only", "both"]
            weights = torch.tensor(
                [probs[m] for m in mode_order],
                dtype=torch.float32,
                device=self.device,
            )
            if float(weights.sum().item()) <= 0:
                weights = torch.ones_like(weights) / len(mode_order)
            pick = int(torch.multinomial(weights, num_samples=1).item())
            return mode_order[pick]
        if fg_text_available:
            return "fg_text_only"
        if fg_proj_ok:
            return "fg_proj_only"
        return "scene_only"

    def process_batch(self, batch):
        """
        Process batch for LiveWorld training with [T, P, R_scene, R_inst] frame order.

        Architecture:
        - Generator input: [noisy_target, preceding, reference_scene, reference_instance]
        - State Adapter input: [target_scene_proj, preceding_scene_proj] (T+P frames)
        - State Adapter hints added to FIRST T+P tokens (no offset with [T, P, R] order)

        Stage behaviour:
        - "state_adapter"/"controlnet": ONLY target scene projection, no preceding, no reference
        - "lora"/"all"/etc: full [T, P, R] with P1/P9 mode and reference frames
        """
        base_prompts = batch["prompts"]
        batch_size = len(base_prompts)
        is_sp_stage = self.current_stage_name in ("state_adapter", "controlnet")

        # ========== 1. Target latent & scene projection ==========
        target_latent = batch["target_latent"].to(device=self.device, dtype=self.dtype)
        overlay_fg = getattr(self.config, "overlay_fg_on_scene", False)
        if overlay_fg and isinstance(batch.get("target_scene_proj_fg_overlay"), torch.Tensor):
            target_scene_proj = batch["target_scene_proj_fg_overlay"].to(device=self.device, dtype=self.dtype)
        elif getattr(self.config, "aug_proj", False):
            # aug_proj=True: target_scene_proj is the augmented version — use it directly
            target_scene_proj = batch["target_scene_proj"].to(device=self.device, dtype=self.dtype)
        else:
            # aug_proj=False: use non-augmented projection.
            # If _orig latent exists (data made with src_aug=True), use it;
            # otherwise target_scene_proj is already non-augmented (src_aug=False).
            orig = batch.get("target_scene_proj_orig")
            if isinstance(orig, torch.Tensor):
                target_scene_proj = orig.to(device=self.device, dtype=self.dtype)
            else:
                target_scene_proj = batch["target_scene_proj"].to(device=self.device, dtype=self.dtype)

        target_frames = self.config.image_or_video_shape[1]
        target_latent = crop_latent_frames(target_latent, target_frames)
        target_scene_proj = crop_latent_frames(target_scene_proj, target_frames)

        B, T, C, H, W = target_latent.shape

        # ========== 2. Preceding (stage-aware) ==========
        if is_sp_stage:
            # State Adapter stage: no preceding frames — force ControlNet to learn from scene projection
            preceding_latent = torch.zeros(B, 0, C, H, W, device=self.device, dtype=self.dtype)
            preceding_scene_proj = torch.zeros(B, 0, target_scene_proj.shape[2], H, W, device=self.device, dtype=self.dtype)
        else:
            # P1/P9 mode selection
            p1_prob = getattr(self.config, "p1_prob", 0.3)
            use_p1 = torch.rand(1).item() < p1_prob
            suffix = "_1" if use_p1 else "_9"
            preceding_latent = batch[f"preceding_latent{suffix}"].to(device=self.device, dtype=self.dtype)
            # When online_fg_aug is on, always load plain scene proj
            if overlay_fg and not online_fg_aug and isinstance(batch.get(f"preceding_scene_proj_fg_overlay{suffix}"), torch.Tensor):
                preceding_scene_proj = batch[f"preceding_scene_proj_fg_overlay{suffix}"].to(device=self.device, dtype=self.dtype)
            else:
                preceding_scene_proj = batch[f"preceding_scene_proj{suffix}"].to(device=self.device, dtype=self.dtype)

        # ========== 3. Reference (stage-aware) ==========
        if is_sp_stage:
            # State Adapter stage: no reference frames
            reference_scene_latent = torch.zeros(B, 0, C, H, W, device=self.device, dtype=self.dtype)
            reference_instance_latent = torch.zeros(B, 0, C, H, W, device=self.device, dtype=self.dtype)
            reference_latent = torch.zeros(B, 0, C, H, W, device=self.device, dtype=self.dtype)
        else:
            separate_ref_fg_bg = getattr(self.config, "separate_ref_fg_bg", True)
            if separate_ref_fg_bg:
                ref_scene_key = "reference_scene_latent" if getattr(self.config, "aug_scene_ref", False) else "reference_scene_latent_orig"
                reference_scene_latent = self._process_reference_latent(
                    batch, batch_size, target_latent, latent_key=ref_scene_key
                )
                reference_instance_latent = self._sample_instance_reference_latent(
                    batch, batch_size, target_latent
                )
                reference_latent = torch.cat([reference_scene_latent, reference_instance_latent], dim=1)
            else:
                reference_latent = self._process_reference_latent(
                    batch, batch_size, target_latent, latent_key="reference_latent"
                )
                reference_scene_latent = reference_latent
                reference_instance_latent = torch.zeros(B, 0, C, H, W, device=self.device, dtype=self.dtype)

        # ========== 4. Text prompts (fg text dropout) ==========
        use_split_prompts = (
            "scene_prompts" in batch
            or "fg_prompts" in batch
            or "has_fg" in batch
            or bool(getattr(self.config, "use_scene_fg_prompt", False))
        )
        scene_prompts = batch.get("scene_prompts") or base_prompts
        fg_prompts = batch.get("fg_prompts") or [""] * batch_size
        has_fg = batch.get("has_fg", None)
        if isinstance(has_fg, torch.Tensor):
            has_fg_list = [bool(x) for x in has_fg.cpu().tolist()]
        elif has_fg is None:
            has_fg_list = [True] * batch_size
        else:
            has_fg_list = [bool(x) for x in has_fg]
        
        # Determine fg_proj availability (including State Adapter stage target_fg path)
        num_pre = preceding_latent.shape[1]
        preceding_fg_key = f"preceding_proj_fg{suffix}" if not is_sp_stage else None
        fg_proj_available = (
            isinstance(batch.get("target_proj_fg"), torch.Tensor)
            and (num_pre == 0 or isinstance(batch.get(preceding_fg_key), torch.Tensor))
        )

        fg_modes = [
            self._select_fg_mode(has_fg_list[idx], fg_prompts[idx], fg_proj_available)
            for idx in range(batch_size)
        ]

        if use_split_prompts:
            text_prompts = []
            for idx in range(batch_size):
                scene_text = scene_prompts[idx]
                fg_text = fg_prompts[idx].strip()
                mode = fg_modes[idx]
                if mode in ("fg_text_only", "both") and fg_text:
                    text_prompts.append(f"scene: {scene_text}; foreground: {fg_text}")
                else:
                    text_prompts.append(f"scene: {scene_text}")
        else:
            text_prompts = base_prompts

        # ========== 5. Foreground projection (always concat to scene proj, 16ch+16ch=32ch) ==========
        if fg_proj_available:
            target_proj_fg = batch["target_proj_fg"].to(device=self.device, dtype=self.dtype)
            target_proj_fg = crop_latent_frames(target_proj_fg, target_frames)

            if num_pre > 0:
                preceding_proj_fg = batch[preceding_fg_key].to(device=self.device, dtype=self.dtype)
            else:
                preceding_proj_fg = torch.zeros_like(preceding_scene_proj)

            fg_proj_keep = torch.tensor(
                [1.0 if mode in ("fg_proj_only", "both") else 0.0 for mode in fg_modes],
                device=target_proj_fg.device, dtype=target_proj_fg.dtype,
            ).view(batch_size, 1, 1, 1, 1)
            target_proj_fg = target_proj_fg * fg_proj_keep
            if num_pre > 0:
                preceding_proj_fg = preceding_proj_fg * fg_proj_keep
        else:
            target_proj_fg = torch.zeros_like(target_scene_proj)
            preceding_proj_fg = torch.zeros_like(preceding_scene_proj)

        target_scene_proj = torch.cat([target_scene_proj, target_proj_fg], dim=2)
        preceding_scene_proj = torch.cat([preceding_scene_proj, preceding_proj_fg], dim=2)

        # ========== 6. Frame counts ==========
        num_ref_scene = reference_scene_latent.shape[1]
        num_ref_inst = reference_instance_latent.shape[1]
        num_ref = num_ref_scene + num_ref_inst
        num_pre = preceding_latent.shape[1]
        num_target = target_latent.shape[1]

        # ========== 7. Encode text & I2V conditioning ==========
        with torch.no_grad():
            conditional_dict = self.text_encoder(text_prompts=text_prompts)
            conditional_dict = {k: v.detach() for k, v in conditional_dict.items()}

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.text_encoder(text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            total_frames = num_ref + num_pre + num_target
            if self.config.i2v:
                img = batch["img"].to(device=self.device, dtype=self.dtype)
                clip_fea = self.image_encoder(img)
                ys = []
                for bs in range(batch_size):
                    y = self.vae.run_vae_encoder(img[bs], new_target_video_length=(total_frames - 1) * 4 + 1)
                    ys.append(y.unsqueeze(0))
                y = torch.cat(ys, dim=0)
            else:
                clip_fea = None
                y = None

        # ========== 8. State Adapter context ==========
        sp_context = self._prepare_sp_context(
            target_scene_proj, preceding_scene_proj, batch_size
        )

        return {
            "target_latent": target_latent,
            "preceding_latent": preceding_latent,
            "reference_latent": reference_latent,
            "reference_scene_latent": reference_scene_latent,
            "reference_instance_latent": reference_instance_latent,
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
            "clip_fea": clip_fea,
            "y": y,
            "sp_context": sp_context,
            "sp_context_scale": 1.0,
            "sp_hint_offset": 0,
            "num_ref": num_ref,
            "num_ref_scene": num_ref_scene,
            "num_ref_inst": num_ref_inst,
            "num_pre": num_pre,
            "num_target": num_target,
        }

    def _prepare_sp_context(
        self,
        target_scene_proj: torch.Tensor,
        preceding_scene_proj: torch.Tensor,
        batch_size: int
    ):
        """
        Prepare State Adapter context from LiveWorld scene projections with [T, P] order.

        Concatenates [target_scene_proj, preceding_scene_proj] along frame dimension.
        With [T, P, R] main frame order, State Adapter hints (T+P) are added to first T+P tokens.

        Args:
            target_scene_proj: [B, T, C, H, W] - scene projection for target views
            preceding_scene_proj: [B, P, C, H, W] - scene projection for preceding views
            batch_size: number of samples in batch

        Returns:
            sp_context: list of [C, T+P, H, W] tensors
        """
        # Concatenate along frame dimension: [B, T+P, C, H, W]
        # Order matches [T, P, R] frame order (target first, then preceding)
        combined = torch.cat([target_scene_proj, preceding_scene_proj], dim=1)

        # Permute to [B, C, T+P, H, W]
        combined = combined.permute(0, 2, 1, 3, 4)

        # Convert to list of [C, T+P, H, W] (model expects this format)
        sp_context = [combined[i] for i in range(batch_size)]

        return sp_context

    def _process_reference_latent(
        self,
        batch,
        batch_size: int,
        target_latent: torch.Tensor,
        latent_key: str = "reference_scene_latent",
    ) -> torch.Tensor:
        """
        Build reference scene latent from full-loaded frames (no padding in dataset).

        Input from dataloader:
        - list of [R_i, C, H, W] tensors (full load), or
        - legacy padded tensor [B, R, C, H, W].

        Output: [B, R, C, H, W] with a consistent R across batch.
        MAX_REF_SCENE_FRAMES is applied here (process_batch), not in dataset.
        """
        ref_data = batch.get(latent_key, None)
        if ref_data is None:
            ref_data = batch.get("reference_latent", None)

        if hasattr(self.config, "MAX_REF_SCENE_FRAMES"):
            max_ref_scene_frames = getattr(self.config, "MAX_REF_SCENE_FRAMES")
        elif hasattr(self.config, "max_ref_scene_frames"):
            max_ref_scene_frames = getattr(self.config, "max_ref_scene_frames")
        else:
            max_ref_scene_frames = 7

        strategy = getattr(self.config, "ref_sample_strategy", "none")

        # Legacy padded tensor path
        if isinstance(ref_data, torch.Tensor):
            ref_padded = ref_data.to(device=self.device, dtype=self.dtype)
            ref_counts = batch.get("reference_scene_count", None)
            if ref_counts is None:
                ref_counts = batch.get("reference_count", None)
            if ref_counts is None:
                ref_counts = torch.full(
                    (batch_size,), ref_padded.shape[1], device=ref_padded.device
                )
            if max_ref_scene_frames is not None and max_ref_scene_frames > 0:
                ref_counts = torch.clamp(ref_counts, max=max_ref_scene_frames)

            min_count = int(ref_counts.min().item()) if batch_size > 0 else 0

            if strategy == "random" and min_count > 0:
                num_ref = random.randint(0, min_count)
            else:
                num_ref = min_count

            if self.current_stage_name == "state_adapter":
                num_ref = 0

            if num_ref == 0:
                C = target_latent.shape[2]
                H = target_latent.shape[3]
                W = target_latent.shape[4]
                return torch.zeros(batch_size, 0, C, H, W, device=self.device, dtype=self.dtype)

            return ref_padded[:, :num_ref, :, :, :]

        # List-of-tensors path (full load)
        C = target_latent.shape[2]
        H = target_latent.shape[3]
        W = target_latent.shape[4]

        if not isinstance(ref_data, list):
            return torch.zeros(batch_size, 0, C, H, W, device=self.device, dtype=self.dtype)

        frames_list = []
        counts = []
        for b_idx in range(batch_size):
            frames = ref_data[b_idx] if b_idx < len(ref_data) else None
            if not isinstance(frames, torch.Tensor) or frames.numel() == 0:
                frames = torch.zeros(0, C, H, W, device=self.device, dtype=self.dtype)
            else:
                if frames.ndim == 3:
                    frames = frames.unsqueeze(0)
                frames = frames.to(device=self.device, dtype=self.dtype)
                if max_ref_scene_frames is not None and max_ref_scene_frames > 0 and frames.shape[0] > max_ref_scene_frames:
                    frames = frames[:max_ref_scene_frames]
            frames_list.append(frames)
            counts.append(frames.shape[0])

        min_count = min(counts) if counts else 0
        if strategy == "random" and min_count > 0:
            num_ref = random.randint(0, min_count)
        else:
            num_ref = min_count

        if self.current_stage_name == "state_adapter":
            num_ref = 0

        if num_ref == 0:
            return torch.zeros(batch_size, 0, C, H, W, device=self.device, dtype=self.dtype)

        selected = [frames[:num_ref] for frames in frames_list]
        return torch.stack(selected, dim=0)

    def _sample_instance_reference_latent(
        self,
        batch,
        batch_size: int,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample one frame per instance and keep all instances.

        Returns:
            Tensor [B, N_inst_max, C, H, W], zero-padded to max instances in batch.
        """
        instance_lists = batch.get("reference_instance_latents", None)
        if instance_lists is None:
            instance_lists = [[] for _ in range(batch_size)]

        C = target_latent.shape[2]
        H = target_latent.shape[3]
        W = target_latent.shape[4]

        sampled_per_sample = []
        counts = []

        for b_idx in range(batch_size):
            per_sample = instance_lists[b_idx] or []
            selected = []
            for frames in per_sample:
                if not isinstance(frames, torch.Tensor):
                    continue
                frames = frames.to(device=self.device, dtype=self.dtype)
                if frames.ndim == 3:
                    frames = frames.unsqueeze(0)
                if frames.shape[0] == 0:
                    continue
                pick = int(torch.randint(0, frames.shape[0], (1,), device=frames.device).item())
                selected.append(frames[pick])
            if selected:
                selected_tensor = torch.stack(selected, dim=0)
            else:
                selected_tensor = torch.zeros(0, C, H, W, device=self.device, dtype=self.dtype)
            sampled_per_sample.append(selected_tensor)
            counts.append(selected_tensor.shape[0])

        max_inst = max(counts) if counts else 0
        if max_inst == 0:
            return torch.zeros(batch_size, 0, C, H, W, device=self.device, dtype=self.dtype)

        padded = []
        for inst_tensor in sampled_per_sample:
            if inst_tensor.shape[0] < max_inst:
                pad = torch.zeros(
                    max_inst - inst_tensor.shape[0], C, H, W,
                    device=self.device, dtype=self.dtype,
                )
                inst_tensor = torch.cat([inst_tensor, pad], dim=0)
            padded.append(inst_tensor)

        return torch.stack(padded, dim=0)
    
    def generator_fm_loss(
        self,
        target_latent,  # [B, T, C, H, W]
        preceding_latent,  # [B, P, C, H, W]
        reference_latent,  # [B, R, C, H, W]
        conditional_dict: dict,
        clip_fea=None,
        y=None,
        sp_context=None,
        sp_context_scale=1.0,
        sp_hint_offset=0,
        num_ref=0,
        num_ref_scene=0,
        num_ref_inst=0,
        num_pre=0,
        num_target=0,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute flow matching loss for LiveWorld with [T, P, R] frame order.

        Generator input: [noisy_target_latent, preceding_latent, reference_latent] concat along frame dim
        State Adapter input: [target_scene_proj, preceding_scene_proj] (T+P frames)
        State Adapter hints are added to FIRST T+P tokens (no offset since T and P are first)

        Loss is computed only on target frames (first T frames in output).

        Preceding Frame Augmentation (LiveWorld paper Appendix 6):
        - Add random noise with timestep in [0, 50] to preceding frames
        - This improves robustness during autoregressive inference
        """
        B, T, C, H, W = target_latent.shape
        P = preceding_latent.shape[1]  # Number of preceding frames
        num_denoising_steps = len(self.denoising_step_list)

        # ========== Preceding Frame Augmentation ==========
        # Add random noise with timestep in [0, 50] to preceding frames
        # (LiveWorld paper Appendix 6: More Implementation Details)
        PRECEDING_NOISE_MAX_TIMESTEP = 400
        if P > 1:  # Multiple preceding frames - add noise augmentation
            # Sample random timestep in [0, 50] for each preceding frame
            timestep_preceding = torch.randint(
                0, PRECEDING_NOISE_MAX_TIMESTEP + 1,
                (B, P), device=self.device, dtype=torch.float32
            )
            # Generate noise for preceding frames
            noise_preceding = torch.randn((B, P, C, H, W), device=self.device, dtype=self.dtype)
            # Add noise using flow matching scheduler
            noisy_preceding = self.scheduler.add_noise(
                preceding_latent.flatten(0, 1),
                noise_preceding.flatten(0, 1),
                timestep_preceding.flatten(0, 1)
            ).unflatten(0, (B, P))
        elif P == 1:
            # Single preceding frame (first generated frame) - no noise, timestep = 0
            noisy_preceding = preceding_latent
            timestep_preceding = torch.zeros((B, 1), device=self.device, dtype=torch.float32)
        else:
            # No preceding frames (P == 0, stage 1 first iteration)
            noisy_preceding = preceding_latent
            timestep_preceding = torch.zeros((B, 0), device=self.device, dtype=torch.float32)

        # ========== Target Frame Noise ==========
        # Sample timesteps for target frames only
        if self.config.generator_task == "bidirectional_video":
            index = torch.randint(0, num_denoising_steps, [B, T], device=self.device)
            timestep_target = self.denoising_step_list[index]  # [B, T]
            task_type = self.config.generator_task if T != 1 else "image"
            timestep_target = process_timestep(timestep_target, task_type=task_type, num_frame_per_block=self.num_frame_per_block)

            # Add noise to target frames
            noise = torch.randn((B, T, C, H, W), device=self.device, dtype=self.dtype)
            noisy_target = self.scheduler.add_noise(
                target_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_target.flatten(0, 1)
            ).unflatten(0, (B, T))
        else:
            # Causal mode
            noisy_target, noise, timestep_target = diffusion_forcing_util(
                image_or_video_shape=target_latent.shape,
                target_latent=target_latent,
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                device=self.device,
                dtype=self.dtype,
                generator_task=self.config.generator_task,
                num_frame_per_block=self.num_frame_per_block
            )

        # Concatenate [noisy_target, noisy_preceding, reference_scene, reference_instance] along frame dimension
        # Frame order: [T, P, R_scene, R_inst] - target frames first for State Adapter hint injection
        # All are [B, F_i, C, H, W]
        combined_input = torch.cat([noisy_target, noisy_preceding, reference_latent], dim=1)  # [B, T+P+R, C, H, W]

        # Create per-frame timestep: T frames get sampled timestep, P frames get augmented timestep, R frames get t=0
        # With [T, P, R_scene, R_inst] order, T frames are FIRST, then P frames
        total_frames = num_target + num_pre + num_ref
        timestep_full = torch.zeros((B, total_frames), device=self.device, dtype=timestep_target.dtype)
        timestep_full[:, :num_target] = timestep_target  # Target frames get sampled timestep
        timestep_full[:, num_target:num_target + num_pre] = timestep_preceding  # Preceding frames get augmented timestep [0, 50]

        # Forward pass through generator
        # Pass segment counts for segment-aware RoPE: [T, P, R_scene, R_inst] frame order
        flow_pred_full, pred_full = self.generator(
            noisy_image_or_video=combined_input,
            timestep=timestep_full,  # [B, R+P+T] - per-frame timestep
            context=conditional_dict['prompt_embeds'],
            clip_fea=clip_fea,
            y=y,
            sp_context=sp_context,
            sp_context_scale=sp_context_scale,
            sp_hint_offset=sp_hint_offset,
            num_t=num_target,  # For segment-aware RoPE
            num_p=num_pre,     # For segment-aware RoPE
            num_r=num_ref,     # For segment-aware RoPE
            num_r_scene=num_ref_scene,
            num_r_inst=num_ref_inst,
        )
        
        # Extract flow prediction for target frames only (FIRST T frames with [T, P, R_scene, R_inst] order)
        # flow_pred_full shape: [B, T+P+R, C, H, W]
        flow_pred_target = flow_pred_full[:, :num_target, :, :, :]

        # Compute denoising loss only on target frames
        denoising_loss = self.denoising_loss_func(
            x=target_latent.flatten(0, 1),
            noise=noise.flatten(0, 1),
            flow_pred=flow_pred_target.flatten(0, 1)
        )

        # Log dict - extract target frames from beginning with [T, P, R_scene, R_inst] order
        # Include preceding frame augmentation info for debugging
        preceding_timestep_info = f", P_timestep_range=[0,{PRECEDING_NOISE_MAX_TIMESTEP}]" if P > 0 else ""
        generator_log_dict = {
            "generator_fm_noisy_input": noisy_target,
            "generator_fm_pred": pred_full[:, :num_target, :, :, :] if pred_full is not None else None,
            "timestep": timestep_target,
            "noise": noise,
            "debugs": f"T={num_target}, P={num_pre}, R_scene={num_ref_scene}, R_inst={num_ref_inst} (frame order: [T, P, R_scene, R_inst]{preceding_timestep_info})"
        }

        return denoising_loss, generator_log_dict

    # ==================== Validation ====================

    def validate(
        self,
        step: int,
        output_path: str,
        is_main_process: bool,
        wandb_loss_dict: Optional[dict] = None,
        log_to_wandb: bool = True,
    ) -> Optional[dict]:
        """
        Run LiveWorld validation inference with scene projection conditioning.

        Args:
            step: Current training step
            output_path: Path to save validation outputs
            is_main_process: Whether this is the main process
            wandb_loss_dict: Dictionary to update with wandb logging info
            log_to_wandb: Whether to log to wandb

        Returns:
            Updated wandb_loss_dict or None
        """
        val_input_base_path = getattr(self.config, "VAL_INPUT_BASE_PATH", None)
        val_filename = getattr(self.config, "VAL_FILENAME", None)

        if val_input_base_path is None or val_filename is None:
            print(f"[VALIDATION] Skipping validation: VAL_INPUT_BASE_PATH={val_input_base_path}, VAL_FILENAME={val_filename}")
            return wandb_loss_dict

        filename_list = [f.strip() for f in val_filename.split(',')]
        infer_steps = getattr(self.config, "VAL_INFER_STEPS", 50)
        num_latent_frames = getattr(self.config, "VAL_NUM_LATENT_FRAMES", self.config.image_or_video_shape[1])
        wandb_loss_dict = {} if wandb_loss_dict is None else wandb_loss_dict

        # For first validation, only validate 1 sample
        if not hasattr(self, '_has_validated_once'):
            self._has_validated_once = False

        if not self._has_validated_once:
            num_samples_to_validate = getattr(self.config, "VAL_FIRST_NUM_SAMPLES", 1)
            filename_list = filename_list[:num_samples_to_validate]
            self._has_validated_once = True
            first_validation_msg = f" (first validation, using {num_samples_to_validate} sample(s))"
        else:
            first_validation_msg = ""

        # All ranks participate in state_dict collection to avoid FSDP hanging
        generator_state_dict = get_full_state_dict(self.generator)

        if not is_main_process:
            return wandb_loss_dict

        with torch.no_grad():
            # Create eval generator with State Adapter support.
            # NOTE: under FSDP, top-level wrapper may not expose peft_config.
            # Fall back to state-dict key inspection to avoid missing LoRA weights.
            has_lora = (
                hasattr(self.generator, 'peft_config')
                or hasattr(getattr(self.generator, 'module', None), 'peft_config')
                or any("lora_" in k for k in generator_state_dict.keys())
            )
            if has_lora:
                lora_key_count = sum(1 for k in generator_state_dict.keys() if "lora_" in k)
                print(f"[VALIDATION] Detected LoRA state dict ({lora_key_count} lora keys)")
            eval_generator = self._create_eval_generator(
                generator_state_dict=generator_state_dict,
                has_lora=has_lora,
                step=step
            )

            val_videos = []
            val_captions = []

            video_output_dir = os.path.join(output_path, "video")
            os.makedirs(video_output_dir, exist_ok=True)

            # Get eval encoders (non-FSDP, on GPU)
            eval_text_encoder, _ = self._get_eval_encoders()

            print(f"\n{'='*80}")
            print(f"[VALIDATION] Starting validation with {len(filename_list)} samples{first_validation_msg}")
            print(f"[VALIDATION] Videos will be saved to: {video_output_dir}")
            print(f"[VALIDATION] Generating {num_latent_frames} latent frames")
            print(f"{'='*80}\n")

            val_comparison_videos = []

            for idx, current_filename in enumerate(filename_list, 1):
                video, comparison_video, caption = self._validate_sample(
                    eval_generator=eval_generator,
                    eval_text_encoder=eval_text_encoder,
                    val_input_base_path=val_input_base_path,
                    filename=current_filename,
                    num_latent_frames=num_latent_frames,
                    infer_steps=infer_steps,
                    video_output_dir=video_output_dir,
                    step=step,
                    idx=idx,
                    total=len(filename_list)
                )
                if video is not None:
                    val_videos.append(video)
                    val_captions.append(caption)
                if comparison_video is not None:
                    val_comparison_videos.append(comparison_video)

            if log_to_wandb and val_videos:
                val_videos = torch.stack(val_videos, dim=0)
                vis_fps = getattr(self.config, "fps", 16)
                wandb_loss_dict.update({
                    "validation": prepare_for_saving(val_videos, fps=vis_fps, caption=val_captions)
                })
                # Log comparison videos (generated | GT) to wandb
                if val_comparison_videos:
                    val_comparison_videos = torch.stack(val_comparison_videos, dim=0)
                    wandb_loss_dict.update({
                        "validation_comparison": prepare_for_saving(val_comparison_videos, fps=vis_fps, caption=val_captions)
                    })

            print(f"{'='*80}")
            print(f"[VALIDATION] All {len(filename_list)} samples completed")
            print(f"{'='*80}\n")

            # Clean up
            del eval_generator
            torch.cuda.empty_cache()

        return wandb_loss_dict

    def _create_eval_generator(
        self,
        generator_state_dict: dict,
        has_lora: bool,
        step: int
    ):
        """Create eval generator with State Adapter support for LiveWorld validation."""
        # Create fresh generator with State Adapter
        eval_generator = BidirectionalWanWrapperSP(config=self.config, role="generator")

        if has_lora:
            # Apply LoRA only to backbone modules (exclude State Adapter), same as training
            lora_config_dict = dict(self.config.lora_config)
            original_targets = lora_config_dict.get("target_modules", ["q", "k", "v", "o"])
            backbone_modules = []
            for name, _ in eval_generator.named_modules():
                if 'sp_' in name.lower():
                    continue
                for target in original_targets:
                    if name.endswith(f".{target}"):
                        backbone_modules.append(name)
                        break
            if backbone_modules:
                lora_config_dict["target_modules"] = backbone_modules
            lora_config = LoraConfig(**lora_config_dict)
            eval_generator = get_peft_model(eval_generator, lora_config)

            # Set inference mode
            for _, cfg in eval_generator.peft_config.items():
                cfg.inference_mode = True

            # Load LoRA weights.
            # generator_state_dict from a PEFT model's .state_dict() already has
            # adapter name "default" in keys (e.g. lora_A.default.weight).
            # set_peft_model_state_dict expects keys WITHOUT the adapter name
            # (it inserts the adapter name itself), so we must strip it first.
            lora_state_dict = {
                k: v for k, v in generator_state_dict.items()
                if "lora_" in k
            }
            # Strip "default." from adapter keys: lora_A.default.weight -> lora_A.weight
            cleaned_lora_state_dict = {}
            for k, v in lora_state_dict.items():
                cleaned_key = k
                cleaned_key = cleaned_key.replace("._fsdp_wrapped_module", "")
                while cleaned_key.startswith("module."):
                    cleaned_key = cleaned_key[len("module."):]
                while cleaned_key.startswith("model."):
                    cleaned_key = cleaned_key[len("model."):]
                cleaned_key = cleaned_key.replace(".default.", ".")
                cleaned_lora_state_dict[cleaned_key] = v
            result = set_peft_model_state_dict(eval_generator, cleaned_lora_state_dict)
            if result.missing_keys:
                print(f"[VALIDATION] WARNING: LoRA missing keys: {result.missing_keys[:5]}")
            if result.unexpected_keys:
                print(f"[VALIDATION] WARNING: LoRA unexpected keys: {result.unexpected_keys[:5]}")
            print(f"[VALIDATION] Loaded {len(cleaned_lora_state_dict)} LoRA parameters at step {step}")

            # Also load State Adapter weights (not handled by set_peft_model_state_dict)
            # State Adapter is NOT wrapped by PEFT (excluded above), so keys match directly
            # State dict keys vary by setup:
            #   - FSDP: "model.model.sp_blocks.xxx"
            #   - Single GPU PEFT: "base_model.model.model.sp_blocks.xxx"
            sp_keys = {}
            for k, v in generator_state_dict.items():
                if 'sp_' in k.lower() and 'lora_' not in k:
                    # Remove prefix to match StateAdapterWanModel's expected keys
                    clean_key = k
                    clean_key = clean_key.replace("._fsdp_wrapped_module", "")
                    # Handle all possible prefixes (order matters - longest first)
                    for prefix in ['base_model.model.model.', 'model.model.', 'base_model.model.', 'model.']:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    sp_keys[clean_key] = v
            if sp_keys:
                missing, unexpected = eval_generator.model.model.load_state_dict(sp_keys, strict=False)
                sp_missing = [k for k in missing if 'sp_' in k.lower()]
                print(f"[VALIDATION] Loaded {len(sp_keys)} State Adapter parameters at step {step}")
                if sp_missing:
                    print(f"[VALIDATION] WARNING: {len(sp_missing)} State Adapter keys missing: {sp_missing[:5]}")
        else:
            # Load full weights - filter for State Adapter keys
            sp_keys = {k: v for k, v in generator_state_dict.items() if 'sp_' in k.lower()}
            if sp_keys:
                eval_generator.model.load_state_dict(sp_keys, strict=False)
                print(f"[VALIDATION] Loaded {len(sp_keys)} State Adapter parameters at step {step}")

        eval_generator.to(device=self.device, dtype=self.dtype)
        eval_generator.eval()

        return eval_generator

    def _validate_sample(
        self,
        eval_generator,
        eval_text_encoder,
        val_input_base_path: str,
        filename: str,
        num_latent_frames: int,
        infer_steps: int,
        video_output_dir: str,
        step: int,
        idx: int,
        total: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """
        Validate a single LiveWorld sample using the same inference pipeline as infer.py.

        Expected directory structure (same as inference):
        - val_input_base_path/filename/clip.mp4 (source video, first frame used)
        - val_input_base_path/filename/geometry.npz (camera poses)
        - val_input_base_path/filename/clip.txt (prompt, optional)

        Returns:
            Tuple of (video_tensor, comparison_tensor, caption_string) or (None, None, "") if failed
            - video_tensor: generated video [T, C, H, W] in [-1, 1] range
            - comparison_tensor: generated|GT side-by-side [T, C, H, 2W] in [-1, 1] range (or None)
            - caption_string: caption for wandb
        """
        # Validation parameters from config
        num_frames = getattr(self.config, "VAL_NUM_FRAMES", 81)
        frames_per_iter = getattr(self.config, "VAL_FRAMES_PER_ITER", 13)
        start_frame = getattr(self.config, "VAL_START_FRAME", 0)

        # Load shared inference defaults if specified
        inference_defaults = {}
        defaults_path = getattr(self.config, "backbone_inference_defaults", None)
        if defaults_path and os.path.exists(defaults_path):
            inference_defaults = OmegaConf.to_container(OmegaConf.load(defaults_path))
            print(f"  Loaded inference defaults from: {defaults_path}")

        print(f"[VALIDATION] Processing sample {idx}/{total}: {filename} ({num_frames} frames)")

        data_dir = os.path.join(val_input_base_path, filename)

        if not os.path.isdir(data_dir):
            print(f"  [WARNING] Directory not found: {data_dir}, skipping")
            return None, None, ""

        # =====================================================================
        # LOAD INPUT DATA (same as scripts/infer.py)
        # =====================================================================
        # Calculate video dimensions
        h, w = self.config.image_or_video_shape[-2:]
        h_pixel = h * self.config.vae_stride[1]
        w_pixel = w * self.config.vae_stride[2]

        # Load geometry_poses_c2w from geometry.npz
        geometry_path = os.path.join(data_dir, "geometry.npz")
        if not os.path.exists(geometry_path):
            print(f"  [WARNING] geometry.npz not found: {geometry_path}, skipping")
            return None, None, ""
        geometry_data = np.load(geometry_path)
        geometry_poses_c2w = geometry_data["poses_c2w"].astype(np.float32)
        print(f"  Loaded geometry.npz: {geometry_poses_c2w.shape[0]} frames")

        # Load first_frame from image
        first_frame_image = None
        for candidate in ["input_cropped.png", "first_frame.png", "frame_0.png", "train_target_rgb_frame0.png"]:
            img_path = os.path.join(data_dir, candidate)
            if os.path.exists(img_path):
                first_frame_image = img_path
                break
        if first_frame_image is None:
            print(f"  [WARNING] No first frame image found in {data_dir}, skipping")
            return None, None, ""
        first_frame = load_frame_from_image(first_frame_image, (w_pixel, h_pixel))
        print(f"  Loaded first frame: {first_frame_image}")

        # Load prompt from clip.txt
        prompt_path = os.path.join(data_dir, "clip.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            print(f"  Loaded prompt from: {prompt_path}")
        else:
            prompt = ""
            print("  No prompt file found, using empty prompt")

        # =====================================================================
        # CREATE PIPELINE AND RUN INFERENCE
        # =====================================================================
        pipeline = UnifiedBackbonePipeline(
            config=self.config,
            device=self.device,
            generator=eval_generator,
            vae=self.vae,
            text_encoder=eval_text_encoder,
            dtype=self.dtype,
        )

        # Build options: start from dataclass defaults, apply inference_defaults,
        # then override validation-specific settings.
        # This ensures validation behaves identically to infer.py.
        valid_fields = {f.name for f in dataclasses.fields(BackboneInferenceOptions)}
        filtered_defaults = {k: v for k, v in inference_defaults.items() if k in valid_fields}
        options = BackboneInferenceOptions(**filtered_defaults)
        # Validation-specific overrides
        options.infer_steps = infer_steps
        options.seed = getattr(self.config, "seed", 42)
        options.fps = inference_defaults.get("fps", getattr(self.config, "fps", 16))
        options.cpu_offload = False  # validation runs on GPU already
        # Stage-aware validation: during State Adapter stage, disable reference frames and preceding
        if getattr(self, "current_stage_name", "").lower() in ("state_adapter", "controlnet"):
            options.max_reference_frames = 0
            options.max_preceding_frames_first_iter = 0
            options.max_preceding_frames_other_iter = 0

        try:
            result = run_iterative_inference(
                pipeline=pipeline,
                first_frame=first_frame,
                prompt=prompt,
                num_frames=num_frames,
                frames_per_iter=frames_per_iter,
                options=options,
                first_frame_image=first_frame_image,
                geometry_poses_c2w=geometry_poses_c2w,
                output_paths=None,  # Don't save intermediate outputs
            )

            final_video = result.generated_video
            print(f"  Total generated frames: {len(final_video)}")

            vis_fps = getattr(self.config, "fps", 16)

            # Save full generated video
            video_filename = f"step_{step:06d}_sample_{idx:02d}_{filename}_{num_frames}frames.mp4"
            video_save_path = os.path.join(video_output_dir, video_filename)
            video_numpy = final_video.numpy()
            imageio.mimsave(video_save_path, video_numpy, fps=vis_fps, macro_block_size=None)
            print(f"[VALIDATION] Saved: {video_filename}")

            # Generate comparison video: generated | scene_proj (projection used for State Adapter)
            comparison_subsampled = None

            # Get scene projection video from result (the actual condition used for generation)
            if not result.state.all_used_scene_proj_frames:
                raise RuntimeError("No scene projection frames available for validation comparison.")
            scene_proj_video = np.stack(result.state.all_used_scene_proj_frames, axis=0)
            # Pad or truncate to match generated video length
            if len(scene_proj_video) < len(final_video):
                pad_count = len(final_video) - len(scene_proj_video)
                scene_proj_video = np.concatenate([scene_proj_video, np.tile(scene_proj_video[-1:], (pad_count, 1, 1, 1))], axis=0)
            elif len(scene_proj_video) > len(final_video):
                scene_proj_video = scene_proj_video[:len(final_video)]

            # Create comparison: generated | scene_proj (projection used for State Adapter)
            comparison_video = np.concatenate([video_numpy, scene_proj_video], axis=2)
            comparison_filename = f"step_{step:06d}_sample_{idx:02d}_{filename}_comparison.mp4"
            comparison_save_path = os.path.join(video_output_dir, comparison_filename)
            imageio.mimsave(comparison_save_path, comparison_video, fps=vis_fps, macro_block_size=None)
            print(f"[VALIDATION] Saved comparison (Gen|Proj): {comparison_filename}")

            # Prepare for wandb
            comparison_tensor = torch.from_numpy(comparison_video)
            comp_subsample_indices = list(range(0, len(comparison_tensor), 2))[:33]
            comparison_subsampled = comparison_tensor[comp_subsample_indices]
            comparison_subsampled = comparison_subsampled.permute(0, 3, 1, 2).float() / 127.5 - 1.0

            # Return video in format expected by wandb ([-1, 1] range, [T, C, H, W])
            # Subsample to ~33 frames for wandb display
            subsample_indices = list(range(0, len(final_video), 2))[:33]
            subsampled_video = final_video[subsample_indices]  # [T, H, W, 3]
            # Convert to [T, C, H, W] and normalize to [-1, 1]
            subsampled_video = subsampled_video.permute(0, 3, 1, 2).float() / 127.5 - 1.0
            return subsampled_video, comparison_subsampled, f"{filename}: {num_frames}frames"

        except Exception as e:
            print(f"  [ERROR] Validation failed: {e}")
            traceback.print_exc()
            return None, None, ""




# ============================================================================
# Trainer
# ============================================================================

import gc
import json
import logging
import shutil
import time
from datetime import datetime

import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .dataset import LiveWorldLMDBDataset, liveworld_collate_fn
from .utils import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job, set_seed

class Trainer:
    """
    Flow Matching Trainer.

    Responsibilities:
    - Initialize distributed training environment
    - Manage training loop
    - Handle checkpointing (save/load)
    - Run validation and visualization
    - Manage training stages
    - Update EMA
    - Log to wandb
    """

    def __init__(self, config):
        self.config = config
        self.step = 0

        self._setup_distributed()
        self._setup_logging()
        self._setup_model()
        self._setup_optimizer()
        self._setup_dataloader()
        self._setup_training_params()

    # ==================== Setup Methods ====================

    def _setup_distributed(self):
        """Initialize distributed training environment."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        set_seed(self.config.seed + dist.get_rank())

        self.dtype = torch.bfloat16 if self.config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = self.config.disable_wandb


    def _setup_logging(self):
        """Setup logging directory and wandb."""
        config = self.config
        config.wandb_name = config.logdir.split("/")[-1]

        if self.is_main_process:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            config.logdir = os.path.join(config.logdir, timestamp)
            os.makedirs(config.logdir, exist_ok=True)
            shutil.copy(config.config_path, config.logdir)

        logdir_list = [config.logdir] if self.is_main_process else [None]
        dist.broadcast_object_list(logdir_list, src=0)
        config.logdir = logdir_list[0]

        self.config = config
        self.output_path = config.logdir

        if self.is_main_process and not self.disable_wandb:
            if config.wandb_mode != "offline":
                wandb.login(key=config.wandb_key)
            wandb_name = self.config.logdir.split("/")[-2]
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=wandb_name,
                mode=config.wandb_mode,
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=os.path.join(config.logdir)
            )


    def _setup_model(self):
        """Initialize model based on task type."""
        task_map = {
            "FMWanLiveWorld": FMWanLiveWorld,
        }

        task_class = task_map.get(self.config.task)
        if task_class is None:
            raise ValueError(f"Unknown task: {self.config.task}")

        self.model = task_class(self.config, device=self.device)

        # Apply FSDP wrapping
        if self.world_size > 1:
            # Get min_num_params from config, with sensible defaults for different model sizes
            generator_min_params = getattr(self.config, "generator_fsdp_min_num_params", 1e8)

            # Wrap generator (trainable model)
            self.model.generator = fsdp_wrap(
                self.model.generator,
                sharding_strategy=self.config.sharding_strategy,
                mixed_precision=self.config.mixed_precision,
                wrap_strategy=self.config.generator_fsdp_wrap_strategy,
                min_num_params=generator_min_params,
            )

            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=self.config.sharding_strategy,
                mixed_precision=self.config.mixed_precision,
                wrap_strategy=self.config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=getattr(self.config, "text_encoder_cpu_offload", False)
            )

            # For frozen encoders, check if we should skip FSDP wrapping
            # frozen encoders don't need gradient sharding, so wrapping them with FSDP
            # only adds communication overhead and can cause shape mismatches
            wrap_frozen_encoders = getattr(self.config, "fsdp_wrap_frozen_encoders", False)

            if wrap_frozen_encoders:
                # Legacy mode: wrap encoders with FSDP (may cause issues with large min_num_params)
                encoder_min_params = getattr(self.config, "encoder_fsdp_min_num_params", 1e8)

                if self.config.i2v:
                    self.model.image_encoder = fsdp_wrap(
                        self.model.image_encoder,
                        sharding_strategy=self.config.sharding_strategy,
                        mixed_precision=self.config.mixed_precision,
                        wrap_strategy=self.config.text_encoder_fsdp_wrap_strategy,
                        min_num_params=encoder_min_params,
                        cpu_offload=getattr(self.config, "image_encoder_cpu_offload", False)
                    )
            else:
                # Recommended mode: don't wrap frozen encoders
                # They will be replicated on all GPUs, but this is fine since they're frozen
                print("[FSDP] Frozen image encoder will NOT be wrapped (replicated on all GPUs)")
        else:
            print("Single GPU, no FSDP to avoid error.")

        # Move to device
        self.model.generator = self.model.generator.to(device=self.device, dtype=torch.bfloat16)
        self.model.vae = self.model.vae.to(device=self.device, dtype=torch.bfloat16)
        if not getattr(self.config, "text_encoder_cpu_offload", False):
            self.model.text_encoder = self.model.text_encoder.to(device=self.device, dtype=torch.bfloat16)
        else:
            print("[Device] text_encoder stays on CPU (text_encoder_cpu_offload=True)")

        if hasattr(self.model, "image_encoder"):
            if not getattr(self.config, "image_encoder_cpu_offload", False):
                self.model.image_encoder = self.model.image_encoder.to(device=self.device, dtype=torch.bfloat16)
            else:
                print("[Device] image_encoder stays on CPU (image_encoder_cpu_offload=True)")
        if hasattr(self.model, "action_encoder"):
            self.model.action_encoder = self.model.action_encoder.to(device=self.device, dtype=torch.bfloat16)


    def _setup_optimizer(self):
        """Setup optimizer (placeholder, will be replaced by training stages)."""
        self.use_training_stages = hasattr(self.config, "training_stages_config_dict")

        print("[TRAINING STAGE] Using training stages mechanism")
        print(f"[TRAINING STAGE] Configured stages: {list(self.config.training_stages_config_dict.keys())}")

        # Create placeholder optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters()][:1],
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

        # Create placeholder scheduler
        self.lr_scheduler = None

        self._print_trainable_params()


    def _print_trainable_params(self):
        """Print trainable parameter count."""
        total = 0
        trainable = 0
        for name, param in self.model.generator.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        print(f"\nTrainable params: {trainable} / {total} ({trainable/total:.2%})")


    def _setup_dataloader(self):
        """Setup data loading."""
        config = self.config

        # Determine dataset type
        dataset_type = getattr(config, "dataset_type", None)

        dataset = LiveWorldLMDBDataset(
            config.data_path,
            config,
            max_samples=config.get("num_samples", None),
        )
        self.dataset_name = "LiveWorld"
        self.collate_fn = liveworld_collate_fn

        self.dataset_len = len(dataset)
        print(f"Dataset {self.dataset_name} contains {len(dataset)} samples.")

        # Fix: Only set start_method if not already set
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)

        dataloader_kwargs = {
            "batch_size": config.batch_size,
            "sampler": sampler,
            "num_workers": getattr(config, 'num_workers', 16),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": getattr(config, 'prefetch_factor', 4),
            "worker_init_fn": nymeria_worker_init_fn,
        }

        # Add custom collate function if defined
        if self.collate_fn is not None:
            dataloader_kwargs["collate_fn"] = self.collate_fn

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))

        self.dataloader = cycle(dataloader)

    def _setup_training_params(self):
        """Setup training parameters (logging intervals, EMA, etc.)."""
        config = self.config
        scale_by_batch = getattr(config, 'scale_steps_by_batch_size', False)
        global_batch_size = config.batch_size * self.world_size

        raw_log_iters = getattr(config, "log_iters", 500)
        raw_vis_iters = getattr(config, "vis_iters", 200)
        raw_ema_start_step = getattr(config, "ema_start_step", 0)

        if scale_by_batch:
            self.log_iters = int(raw_log_iters / global_batch_size)
            self.vis_iters = int(raw_vis_iters / global_batch_size)
            self.ema_start_step = int(raw_ema_start_step / global_batch_size)
        else:
            self.log_iters = raw_log_iters
            self.vis_iters = raw_vis_iters
            self.ema_start_step = raw_ema_start_step

        if self.is_main_process:
            self._print_logging_config(config, scale_by_batch, global_batch_size, raw_log_iters, raw_vis_iters, raw_ema_start_step)

        # EMA setup
        self.generator_ema = None
        self.ema_weight = config.ema_weight

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 1)
        self.previous_time = None

    def _print_logging_config(self, config, scale_by_batch, global_batch_size,
                              raw_log_iters, raw_vis_iters, raw_ema_start_step):
        """Print logging configuration."""
        print(f"\n{'='*80}")
        print(f"[LOGGING CONFIG]")
        print(f"  - Batch size per GPU: {config.batch_size}")
        print(f"  - Number of GPUs: {self.world_size}")
        print(f"  - Effective batch size: {global_batch_size}")
        if scale_by_batch:
            print(f"  - Config values represent TOTAL SAMPLES (will be scaled)")
            print(f"  - log_iters: {self.log_iters} steps (from {raw_log_iters} samples)")
            print(f"  - vis_iters: {self.vis_iters} steps (from {raw_vis_iters} samples)")
            print(f"  - ema_start_step: {self.ema_start_step} steps (from {raw_ema_start_step} samples)")
        else:
            print(f"  - Config values represent STEPS directly")
            print(f"  - log_iters: {self.log_iters} steps")
            print(f"  - vis_iters: {self.vis_iters} steps")
            print(f"  - ema_start_step: {self.ema_start_step} steps")
        print(f"{'='*80}\n")

    
    #! ==================== Training ====================

    def train(self):
        """Main training loop."""
        start_step = self.step

        if not hasattr(self, "cumulative_steps"):
            self.switch_or_setup_training_stages()

        self._initialize_ema_if_needed()

        total_train_steps = self.cumulative_steps[-1] if hasattr(self, "cumulative_steps") else self.dataset_len

        pbar = tqdm(
            desc="Training Steps",
            total=total_train_steps,
            dynamic_ncols=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            ncols=120
        )

        while True:
            self.switch_or_setup_training_stages()

            batch = next(self.dataloader)
            # main training
            generator_log_dict = self.train_one_step(batch)

            # ema
            self._update_ema()

            # Save checkpoint
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.log_iters == 0:
                self.save_checkpoint()
                torch.cuda.empty_cache()

            # Logging
            wandb_loss_dict = self._prepare_loss_dict(generator_log_dict)

            # Validation and visualization
            should_validate = self.step % self.vis_iters == 0
            if should_validate:
                # Delegate validation to model (allows task-specific validation)
                self.model.validate(
                    step=self.step,
                    output_path=self.output_path,
                    is_main_process=self.is_main_process,
                    wandb_loss_dict=wandb_loss_dict,
                    log_to_wandb=not self.disable_wandb
                )
                if self.is_main_process and not self.disable_wandb:
                    with torch.no_grad():
                        self.add_visualization(generator_log_dict, wandb_loss_dict)
                    wandb.log(wandb_loss_dict, step=self.step)
                torch.distributed.barrier()

            if not self.disable_wandb and self.is_main_process:
                wandb.log(wandb_loss_dict, step=self.step)

            # Garbage collection
            if self.step % 50 == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            # Log timing
            self._log_timing()

            self.step += 1
            pbar.update(1)

            if self.is_main_process:
                pbar.set_postfix({
                    "FM_Loss": f"{generator_log_dict['denoising_loss'].mean().item():.4f}",
                })

            if self.step >= total_train_steps:
                # Final validation before ending training
                if self.is_main_process:
                    print(f"\n[TRAINING] Running final validation at step {self.step}...")
                final_wandb_loss_dict = self._prepare_loss_dict(generator_log_dict)
                self.model.validate(
                    step=self.step,
                    output_path=self.output_path,
                    is_main_process=self.is_main_process,
                    wandb_loss_dict=final_wandb_loss_dict,
                    log_to_wandb=not self.disable_wandb
                )
                if self.is_main_process and not self.disable_wandb:
                    with torch.no_grad():
                        self.add_visualization(generator_log_dict, final_wandb_loss_dict)
                    wandb.log(final_wandb_loss_dict, step=self.step)
                torch.distributed.barrier()

                torch.cuda.empty_cache()
                self.save_checkpoint()
                torch.cuda.empty_cache()
                break
            

    def train_one_step(self, batch: dict) -> dict:
        """Execute one training step."""
        self.generator_optimizer.zero_grad(set_to_none=True)

        batch_dict = self.model.process_batch(batch)
        denoising_loss, generator_log_dict = self.model.generator_fm_loss(**batch_dict)
        denoising_loss.backward()

        params = (p for p in self.model.generator.parameters() if p.requires_grad)
        generator_grad_norm = clip_grad_norm_(params, max_norm=self.max_grad_norm_generator, norm_type=2.0)

        self.generator_optimizer.step()

        # Update learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        generator_log_dict.update({
            "denoising_loss": denoising_loss,
            "generator_grad_norm": generator_grad_norm,
            "text_prompts": batch["prompts"],
            "target_latent": batch["target_latent"].to(self.device, self.dtype),
        })

        return generator_log_dict


    def _prepare_loss_dict(self, generator_log_dict: dict) -> dict:
        """Prepare loss dictionary for logging."""
        wandb_loss_dict = {}
        if self.is_main_process:
            wandb_loss_dict.update({
                "denoising_loss": generator_log_dict["denoising_loss"].mean().item(),
                "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
            })
            if hasattr(self, "current_stage_idx"):
                wandb_loss_dict["training_stage_idx"] = self.current_stage_idx
                wandb_loss_dict["training_stage_name"] = self.stage_names[self.current_stage_idx]

            # Log current learning rate
            if self.lr_scheduler is not None:
                current_lr = self.lr_scheduler.get_last_lr()[0]
                wandb_loss_dict["learning_rate"] = current_lr
        return wandb_loss_dict


    def _initialize_ema_if_needed(self):
        """Initialize EMA if conditions are met."""
        if (self.step >= self.ema_start_step) and (self.generator_ema is None) and \
           (self.ema_weight is not None) and (self.ema_weight > 0):
            trainable_only = getattr(self.config, 'ema_trainable_only', True)
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight, trainable_only=trainable_only)
            if self.is_main_process:
                print(f"[EMA] Initialized with weight {self.ema_weight}, trainable_only={trainable_only}")


    def _update_ema(self):
        """Update EMA if enabled."""
        if (self.step >= self.ema_start_step) and (self.generator_ema is None) and (self.config.ema_weight > 0):
            trainable_only = getattr(self.config, 'ema_trainable_only', True)
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight, trainable_only=trainable_only)

        ema_update_interval = getattr(self.config, 'ema_update_interval', 1)
        if self.generator_ema is not None and self.step % ema_update_interval == 0:
            self.generator_ema.update(self.model.generator)

    
    def _log_timing(self):
        """Log per-iteration timing."""
        if self.is_main_process:
            current_time = time.time()
            if self.previous_time is None:
                self.previous_time = current_time
            else:
                if not self.disable_wandb:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                self.previous_time = current_time

    # ==================== Checkpoint ====================

    def save_checkpoint(self):
        """Save model checkpoint.

        If training_stages_config_dict is defined, saves all parameters for all stages
        (e.g., both state_adapter and lora parameters regardless of current training stage).
        This ensures checkpoints are complete and can resume from any stage.
        """
        print("Start gathering distributed model states...")

        has_lora = hasattr(self.model.generator, 'peft_config')
        training_stages = getattr(self.config, "training_stages_config_dict", None)

        if training_stages:
            # Save all parameters for all stages defined in training_stages_config_dict
            generator_state_dict = self._gather_stage_parameters(training_stages, has_lora)
        elif has_lora:
            generator_state_dict = get_peft_model_state_dict(self.model.generator)
            print(f"Saving LoRA parameters: {len(generator_state_dict)} parameters")
        else:
            generator_state_dict = fsdp_state_dict(self.model.generator)

        if self.generator_ema is not None:
            generator_ema_state_dict = self.generator_ema.state_dict()
            generator_ema_state_dict = {
                k.replace("._fsdp_wrapped_module", ""): v
                for k, v in generator_ema_state_dict.items()
            }
            if generator_ema_state_dict:
                first_key = next(iter(generator_ema_state_dict.keys()))
                if not first_key.startswith("model."):
                    generator_ema_state_dict = {f"model.{k}": v for k, v in generator_ema_state_dict.items()}
        else:
            generator_ema_state_dict = {}

        if self.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": generator_ema_state_dict,
            }
        else:
            state_dict = {"generator": generator_state_dict}

        if has_lora:
            state_dict["is_lora"] = True
            state_dict["lora_config"] = self.model.generator.peft_config

        if training_stages:
            state_dict["training_stages"] = list(training_stages.keys())

        if self.is_main_process:
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    def _gather_stage_parameters(self, training_stages: dict, has_lora: bool) -> dict:
        """Gather parameters for all training stages.

        Args:
            training_stages: Dict of stage_name -> [max_steps, lr]
            has_lora: Whether model has LoRA

        Returns:
            Combined state dict with all stage parameters
        """
        stage_names = list(training_stages.keys())
        print(f"Saving parameters for stages: {stage_names}")

        combined_state_dict = {}

        # Collect full state dict first (for non-LoRA parameters like state_adapter)
        full_state_dict = fsdp_state_dict(self.model.generator)

        for stage_name in stage_names:
            stage_name_lower = stage_name.lower()

            if stage_name_lower == "lora" and has_lora:
                # Get LoRA parameters
                lora_state_dict = get_peft_model_state_dict(self.model.generator)
                combined_state_dict.update(lora_state_dict)
                print(f"  - {stage_name}: {len(lora_state_dict)} LoRA parameters")

            elif stage_name_lower in ("state_adapter", "controlnet"):
                # Get State Adapter parameters from full state dict
                sp_params = {k: v for k, v in full_state_dict.items() if 'sp_' in k.lower()}
                combined_state_dict.update(sp_params)
                print(f"  - {stage_name}: {len(sp_params)} State Adapter parameters")

            elif stage_name_lower == "control_adapter":
                # Get control_adapter parameters
                adapter_params = {k: v for k, v in full_state_dict.items() if 'control_adapter' in k.lower()}
                combined_state_dict.update(adapter_params)
                print(f"  - {stage_name}: {len(adapter_params)} control_adapter parameters")

            elif stage_name_lower in ("backbone", "backbone_only"):
                # Get backbone parameters (everything except sp_/lora/control_adapter)
                backbone_params = {
                    k: v for k, v in full_state_dict.items()
                    if not any(x in k.lower() for x in ['sp_', 'lora', 'control_adapter'])
                }
                combined_state_dict.update(backbone_params)
                print(f"  - {stage_name}: {len(backbone_params)} backbone parameters")

            elif stage_name_lower in ("all", "full"):
                # Save everything
                combined_state_dict.update(full_state_dict)
                if has_lora:
                    lora_state_dict = get_peft_model_state_dict(self.model.generator)
                    combined_state_dict.update(lora_state_dict)
                print(f"  - {stage_name}: all parameters")

            else:
                # Generic: filter by stage name in parameter key
                stage_params = {k: v for k, v in full_state_dict.items() if stage_name_lower in k.lower()}
                if stage_params:
                    combined_state_dict.update(stage_params)
                    print(f"  - {stage_name}: {len(stage_params)} parameters (matched by name)")
                else:
                    print(f"  - {stage_name}: WARNING - no parameters matched")

        print(f"Total: {len(combined_state_dict)} parameters to save")
        return combined_state_dict

    # ==================== Visualization ====================

    def add_visualization(self, generator_log_dict: dict, wandb_loss_dict: dict):
        """Add training visualization to wandb."""
        generator_fm_pred, target_latent, generator_fm_noisy_input = map(
            lambda x: self.model.vae.decode_to_pixel(x).squeeze(1),
            [
                generator_log_dict['generator_fm_pred'],
                generator_log_dict['target_latent'],
                generator_log_dict['generator_fm_noisy_input'],
            ]
        )

        timesteps = generator_log_dict["timestep"][:10, 0]
        prompts = generator_log_dict['text_prompts'][:10]

        for i in range(len(prompts)):
            prompts[i] = f"{str(timesteps[i].item())}_{prompts[i]}"

        vis_fps = getattr(self.config, "fps", 16)
        wandb_loss_dict.update({
            "generator_fm_pred": prepare_for_saving(generator_fm_pred[:10], fps=vis_fps, caption=prompts),
            "target_latent": prepare_for_saving(target_latent[:10], fps=vis_fps, caption=prompts),
            "generator_fm_noisy_input": prepare_for_saving(generator_fm_noisy_input[:10], fps=vis_fps, caption=prompts),
        })

    # ==================== Training Stages ====================

    def switch_or_setup_training_stages(self):
        """Manage training stages."""
        if not hasattr(self.config, "training_stages_config_dict"):
            return False

        training_stages_config = self.config.training_stages_config_dict

        if not hasattr(self, "current_stage_idx"):
            self._initialize_training_stages(training_stages_config)
            return self._setup_stage(0)

        current_cumulative_step = self.cumulative_steps[self.current_stage_idx]

        if self.step >= current_cumulative_step and self.current_stage_idx < len(self.stage_names) - 1:
            if not self.config.no_save:
                if self.is_main_process:
                    print(f"\n[TRAINING STAGE] Saving checkpoint before switching from stage '{self.stage_names[self.current_stage_idx]}' to next stage...")
                torch.cuda.empty_cache()
                self.save_checkpoint()
                torch.cuda.empty_cache()
                if self.is_main_process:
                    print(f"[TRAINING STAGE] Checkpoint saved at step {self.step}\n")

            self.current_stage_idx += 1
            return self._setup_stage(self.current_stage_idx)

        return False

    def _initialize_training_stages(self, training_stages_config):
        """Initialize training stage tracking."""
        self.current_stage_idx = 0
        self.stage_names = list(training_stages_config.keys())
        self.cumulative_steps = []
        cumulative = 0

        scale_by_batch = getattr(self.config, 'scale_steps_by_batch_size', False)
        global_batch_size = self.config.batch_size * self.world_size

        if scale_by_batch and self.is_main_process:
            print(f"\n{'='*80}")
            print(f"[TRAINING STAGE] Config values represent TOTAL SAMPLES")
            print(f"  - Global batch size: {global_batch_size}")
            print(f"  - Steps will be calculated as: total_samples / global_batch_size")
            print(f"{'='*80}\n")

        for stage_name in self.stage_names:
            stage_config = training_stages_config[stage_name]
            config_value = stage_config[0]

            if scale_by_batch:
                actual_steps = int(config_value / global_batch_size)
            else:
                actual_steps = config_value

            cumulative += actual_steps
            self.cumulative_steps.append(cumulative)

        if self.is_main_process:
            self._print_training_stages(training_stages_config, scale_by_batch, global_batch_size)

    def _print_training_stages(self, training_stages_config, scale_by_batch, global_batch_size):
        """Print training stage configuration."""
        print(f"\n{'='*80}")
        print(f"[TRAINING STAGE] Configuration:")
        print(f"  - Batch size per GPU: {self.config.batch_size}")
        print(f"  - Number of GPUs: {self.world_size}")
        print(f"  - Effective global batch size: {global_batch_size}")
        if scale_by_batch:
            print(f"  - Config interpretation: Values represent TOTAL SAMPLES")
        else:
            print(f"  - Config interpretation: Values represent STEPS")
        print(f"\n[TRAINING STAGE] Stages:")

        cumulative = 0
        for i, stage_name in enumerate(self.stage_names):
            stage_config = training_stages_config[stage_name]
            config_value = stage_config[0]

            if scale_by_batch:
                total_samples = config_value
                actual_steps = int(total_samples / global_batch_size)
            else:
                actual_steps = config_value
                total_samples = config_value * global_batch_size

            cumulative += actual_steps
            print(f"  - {stage_name}: {actual_steps:,} steps (total samples: {total_samples:,}, cumulative steps: {cumulative:,})")

        print(f"{'='*80}\n")

    def _setup_stage(self, stage_idx: int) -> bool:
        """Setup a specific training stage."""
        stage_name = self.stage_names[stage_idx]
        stage_config = self.config.training_stages_config_dict[stage_name]
        max_steps, lr = stage_config[0], stage_config[1]

        scale_by_batch = getattr(self.config, 'scale_steps_by_batch_size', False)
        global_batch_size = self.config.batch_size * self.world_size

        # Calculate actual steps for this stage
        if scale_by_batch:
            stage_steps = int(max_steps / global_batch_size)
        else:
            stage_steps = max_steps

        print(f"\n{'='*80}")
        print(f"[TRAINING STAGE] Setting up stage {stage_idx + 1}/{len(self.stage_names)}: '{stage_name}'")
        print(f"[TRAINING STAGE] Learning rate: {lr}")
        print(f"[TRAINING STAGE] Training steps for this stage: {stage_steps}")
        print(f"[TRAINING STAGE] Will run until global step: {self.cumulative_steps[stage_idx]}")
        print(f"[TRAINING STAGE] Effective batch size: {global_batch_size}")

        # Notify task about stage change (for stage-specific data processing)
        if hasattr(self.model, 'set_training_stage'):
            self.model.set_training_stage(stage_name)

        trainable_params = self.model.get_trainable_parameters(stage=stage_name)

        self.generator_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

        # Setup learning rate scheduler
        scheduler_type = getattr(self.config, 'lr_scheduler_type', 'cosine')
        warmup_steps = getattr(self.config, 'lr_warmup_steps', 0)
        min_lr_ratio = getattr(self.config, 'min_lr_ratio', 0.0)

        if scheduler_type == 'cosine':
            # Create warmup scheduler if needed
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.generator_optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                cosine_scheduler = CosineAnnealingLR(
                    self.generator_optimizer,
                    T_max=stage_steps - warmup_steps,
                    eta_min=lr * min_lr_ratio
                )
                self.lr_scheduler = SequentialLR(
                    self.generator_optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
                print(f"[LR SCHEDULER] Using CosineAnnealingLR with {warmup_steps} warmup steps")
            else:
                self.lr_scheduler = CosineAnnealingLR(
                    self.generator_optimizer,
                    T_max=stage_steps,
                    eta_min=lr * min_lr_ratio
                )
                print(f"[LR SCHEDULER] Using CosineAnnealingLR without warmup")

        elif scheduler_type == 'linear':
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.generator_optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                decay_scheduler = LinearLR(
                    self.generator_optimizer,
                    start_factor=1.0,
                    end_factor=min_lr_ratio,
                    total_iters=stage_steps - warmup_steps
                )
                self.lr_scheduler = SequentialLR(
                    self.generator_optimizer,
                    schedulers=[warmup_scheduler, decay_scheduler],
                    milestones=[warmup_steps]
                )
                print(f"[LR SCHEDULER] Using LinearLR with {warmup_steps} warmup steps")
            else:
                self.lr_scheduler = LinearLR(
                    self.generator_optimizer,
                    start_factor=1.0,
                    end_factor=min_lr_ratio,
                    total_iters=stage_steps
                )
                print(f"[LR SCHEDULER] Using LinearLR without warmup")

        elif scheduler_type == 'constant':
            self.lr_scheduler = None
            print(f"[LR SCHEDULER] Using constant learning rate (no scheduler)")

        else:
            raise ValueError(f"Unknown lr_scheduler_type: {scheduler_type}")

        if self.lr_scheduler is not None:
            print(f"[LR SCHEDULER] Min LR ratio: {min_lr_ratio} (min_lr = {lr * min_lr_ratio:.2e})")

        print(f"{'='*80}\n")

        total_params = sum(p.numel() for p in self.model.generator.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        print(f"[TRAINING STAGE] Trainable params for '{stage_name}': {trainable:,} / {total_params:,} ({trainable/total_params:.2%})")

        if self.is_main_process and not self.disable_wandb:
            wandb.log({
                "training_stage_idx": stage_idx,
                "training_stage_name": stage_name,
                "training_stage_initial_lr": lr,  # Renamed to clarify it's the initial LR
                "training_stage_trainable_params": trainable,
            }, step=self.step)
            # Note: The actual scheduled learning_rate will be logged in train_one_step()

        return True
