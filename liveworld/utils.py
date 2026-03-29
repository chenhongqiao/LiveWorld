from __future__ import annotations
"""
LiveWorld utilities.
"""

# === fsdp ===

# Copyied from https://github.com/Wan-Video/Wan2.1/blob/main/wan/distributed/fsdp.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import pathlib
import types
from functools import partial

import pandas as pd

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    module_to_wrapper=None,
):  
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks if module_to_wrapper is None else module_to_wrapper),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    return model

def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
# === loss ===


from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class DenoisingLoss(ABC):
    """Base class for denoising losses."""

    @abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class X0PredLoss(DenoisingLoss):
    """Loss on x0 prediction."""

    def __call__(self, x: torch.Tensor, x0_pred: torch.Tensor, **_kwargs) -> torch.Tensor:
        return F.mse_loss(x0_pred, x)


class VPredLoss(DenoisingLoss):
    """Loss on v prediction (if used)."""

    def __call__(self, v: torch.Tensor, v_pred: torch.Tensor, **_kwargs) -> torch.Tensor:
        return F.mse_loss(v_pred, v)


class NoisePredLoss(DenoisingLoss):
    """Loss on noise prediction."""

    def __call__(self, noise: torch.Tensor, noise_pred: torch.Tensor, **_kwargs) -> torch.Tensor:
        return F.mse_loss(noise_pred, noise)


class FlowPredLoss(DenoisingLoss):
    """Loss on flow prediction (noise - x0)."""

    def __call__(self, x: torch.Tensor, noise: torch.Tensor, flow_pred: torch.Tensor, **_kwargs) -> torch.Tensor:
        flow_target = noise - x
        return F.mse_loss(flow_pred, flow_target)


_LOSS_MAP = {
    "x0": X0PredLoss,
    "v": VPredLoss,
    "noise": NoisePredLoss,
    "flow": FlowPredLoss,
}


def get_denoising_loss(loss_type: str) -> type[DenoisingLoss]:
    """Return loss class by name."""
    if loss_type not in _LOSS_MAP:
        raise ValueError(f"Unknown denoising loss type: {loss_type}")
    return _LOSS_MAP[loss_type]

# === scheduler ===

from abc import abstractmethod, ABC
import torch


class SchedulerInterface(ABC):
    """
    Base class for diffusion noise schedule.
    """
    alphas_cumprod: torch.Tensor  # [T], alphas for defining the noise schedule

    @abstractmethod
    def add_noise(
        self, clean_latent: torch.Tensor,
        noise: torch.Tensor, timestep: torch.Tensor
    ):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """
        pass

    def convert_x0_to_noise(
        self, x0: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's x0 prediction to noise predidction.
        x0: the predicted clean data with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(
            lambda x: x.double().to(x0.device), [x0, xt,
                                                 self.alphas_cumprod]
        )

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t **
                      (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(
        self, noise: torch.Tensor, xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's noise prediction to x0 predidction.
        noise: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(
            lambda x: x.double().to(noise.device), [noise, xt, self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t **
                   (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(
        self, 
        velocity: torch.Tensor, 
        xt: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's velocity prediction to x0 predidction.
        velocity: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        v = sqrt(alpha_t) * noise - sqrt(beta_t) x0
        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t)
        given v, x_t, we have
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        see derivations https://chatgpt.com/share/679fb6c8-3a30-8008-9b0e-d1ae892dac56
        """
        # use higher precision for calculations
        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(
            lambda x: x.double().to(velocity.device), [velocity, xt,
                                                       self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t ** 0.5) * xt - (beta_prod_t ** 0.5) * velocity
        return x0_pred.to(original_dtype)


class FlowMatchScheduler():
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / \
            (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (
                self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """
        Input:
            - timestep: the timestep with shape [B*T]
        Output: the corresponding weighting [B*T]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


import math
from typing import Optional, Literal

import torch
import torch.nn as nn



# === video_encode ===

"""Video encoding utilities with explicit H.264 settings."""

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch


def _to_uint8_rgb_array(frames: Union[np.ndarray, torch.Tensor, Iterable[np.ndarray]]) -> np.ndarray:
    """Normalize input frames to uint8 RGB array with shape [T, H, W, 3]."""
    if isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().numpy()
    elif isinstance(frames, np.ndarray):
        arr = frames
    else:
        arr = np.stack(list(frames), axis=0)

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [T, H, W, 3], got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def save_video_h264(
    path: Union[str, Path],
    frames: Union[np.ndarray, torch.Tensor, Iterable[np.ndarray]],
    fps: float = 16.0,
) -> None:
    """Save frames as MP4 (H.264, yuv420p, faststart) using FFmpeg."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = _to_uint8_rgb_array(frames)
    if arr.shape[0] == 0:
        return

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH; cannot encode H.264 output video")

    t, h, w, _ = arr.shape
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        for i in range(t):
            proc.stdin.write(arr[i].tobytes())
        proc.stdin.close()
        assert proc.stderr is not None
        stderr = proc.stderr.read()
        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg H.264 encode failed for {out}: {err}")

# === lora_utils ===

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/bmaltais/kohya_ss

from collections import defaultdict

import torch
import torch.utils.checkpoint
from safetensors.torch import load_file


# =============================================================================
# General LoRA Loader (moved from diffsynth_lora.py)
# =============================================================================

class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(
                    device=self.device, dtype=self.torch_dtype
                )
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(
                    device=self.device, dtype=self.torch_dtype
                )
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = (
                    state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                )
                module.load_state_dict(state_dict)
                updated_num += 1
        print(f"{updated_num} tensors are updated by LoRA.")



def merge_lora(pipeline, lora_path, multiplier, device='cpu', dtype=torch.float32, state_dict=None, transformer_only=False, sub_transformer_name="transformer"):
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if state_dict is None:
        state_dict = load_file(lora_path)
    else:
        state_dict = state_dict
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        if "diffusion_model" in key:
            key = key.replace("diffusion_model.", "lora_unet__")
            key = key.replace("blocks.", "blocks_")
            key = key.replace(".self_attn.", "_self_attn_")
            key = key.replace(".cross_attn.", "_cross_attn_")
            key = key.replace(".ffn.", "_ffn_")
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    sequential_cpu_offload_flag = False
    if pipeline.transformer.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True
        offload_device = pipeline._offload_device

    for layer, elems in updates.items():

        if "lora_te" in layer:
            if transformer_only:
                continue
            else:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            curr_layer = getattr(pipeline, sub_transformer_name)

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                print(f'Error loading layer in front search: {layer}. Try it in back search.')
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                if "lora_te" in layer:
                    if transformer_only:
                        continue
                    else:
                        layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                        curr_layer = pipeline.text_encoder
                else:
                    layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
                    curr_layer = getattr(pipeline, sub_transformer_name)

                len_layer_infos = len(layer_infos)
                start_index     = 0 if len_layer_infos >= 1 and len(layer_infos[0]) > 0 else 1
                end_indx        = len_layer_infos

                error_flag      = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            print(f'Error loading layer in back search: {layer}')
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__("_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue
        try:
            origin_dtype = curr_layer.weight.data.dtype
            origin_device = curr_layer.weight.data.device
        except:
            breakpoint()

        curr_layer = curr_layer.to(device, dtype)
        weight_up = elems['lora_up.weight'].to(device, dtype)
        weight_down = elems['lora_down.weight'].to(device, dtype)
        
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
        curr_layer = curr_layer.to(origin_device, origin_dtype)

    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=offload_device)
    return pipeline


# TODO: Refactor with merge_lora.
def unmerge_lora(pipeline, lora_path, multiplier=1, device="cpu", dtype=torch.float32, sub_transformer_name="transformer"):
    """Unmerge state_dict in LoRANetwork from the pipeline in diffusers."""
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    state_dict = load_file(lora_path)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        if "diffusion_model" in key:
            key = key.replace("diffusion_model.", "lora_unet__")
            key = key.replace("blocks.", "blocks_")
            key = key.replace(".self_attn.", "_self_attn_")
            key = key.replace(".cross_attn.", "_cross_attn_")
            key = key.replace(".ffn.", "_ffn_")
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    sequential_cpu_offload_flag = False
    if pipeline.transformer.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True

    for layer, elems in updates.items():

        if "lora_te" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = getattr(pipeline, sub_transformer_name)

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                print(f'Error loading layer in front search: {layer}. Try it in back search.')
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                if "lora_te" in layer:
                    layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                    curr_layer = pipeline.text_encoder
                else:
                    layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                    curr_layer = getattr(pipeline, sub_transformer_name)
                len_layer_infos = len(layer_infos)

                start_index     = 0 if len_layer_infos >= 1 and len(layer_infos[0]) > 0 else 1
                end_indx        = len_layer_infos

                error_flag      = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            print(f'Error loading layer in back search: {layer}')
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__("_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue

        origin_dtype = curr_layer.weight.data.dtype
        origin_device = curr_layer.weight.data.device

        curr_layer = curr_layer.to(device, dtype)
        weight_up = elems['lora_up.weight'].to(device, dtype)
        weight_down = elems['lora_down.weight'].to(device, dtype)
        
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        if len(weight_up.shape) == 4:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up, weight_down)
        curr_layer = curr_layer.to(origin_device, origin_dtype)

    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=device)
    return pipeline

# === distributed ===

from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    try:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
        ):
            checkpoint = model.state_dict()
    except:
        print("No FSDP env! fall back to normal torch checkpoint")
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=1e8, transformer_module=None, cpu_offload=False):
    if mixed_precision:
        if mixed_precision == "bf16":
            param_dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            param_dtype = torch.float16
        else:
            # Default to bfloat16 if mixed_precision is True (boolean)
            param_dtype = torch.bfloat16

        mixed_precision_policy = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks,
        
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    timeout = timedelta(minutes=120)

    if rank == 0 and backend == "nccl":
        nccl_timeout_env = os.environ.get("NCCL_TIMEOUT", "unset")
        print(f"[dist] NCCL_TIMEOUT env (seen by PyTorch): {nccl_timeout_env}")
        print(f"[dist] init_process_group timeout: {timeout}")

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timeout)
    torch.cuda.set_device(local_rank)


# =============================================================================
# EMA_FSDP: 适配 FSDP 的 EMA 管理器
# - 仅 rank0 维护 shadow（在 CPU），各 rank 同步通过 barrier 协调
# - 支持 trainable_only 模式：只对可训练参数做EMA，大幅减少开销
# - FSDP(FULL_SHARD/SHARD_GRAD_OP)：用 summon_full_params(offload_to_cpu=False)
# - FSDP(NO_SHARD) 或 非 FSDP：直接遍历 module.named_parameters()
# =============================================================================


def _is_dist_init() -> bool:
    """检查是否已初始化分布式环境。"""
    return dist.is_available() and dist.is_initialized()


def _is_fsdp(m: torch.nn.Module) -> bool:
    """是否为 FSDP 包装模块。"""
    return (FSDP is not None) and isinstance(m, FSDP)


def _is_no_shard(m: torch.nn.Module) -> bool:
    """是否为 FSDP 的 NO_SHARD 策略。非 FSDP 返回 False。"""
    if not _is_fsdp(m):
        return False
    # 有些版本属性名不同，这里尽量兼容
    strat = getattr(m, "sharding_strategy", None)
    return (ShardingStrategy is not None) and (strat == ShardingStrategy.NO_SHARD)


def _iter_named_params_full(m: torch.nn.Module):
    """
    统一的参数遍历器：
    - FSDP(FULL_SHARD/SHARD_GRAD_OP)：在 summon_full_params(offload_to_cpu=False) 中遍历底层 module 的参数；
    - FSDP(NO_SHARD) 或 非 FSDP：直接遍历（无需 summon）。
    返回 (name, param) 迭代器，param 为当前设备上的 Tensor（可能在 GPU）。
    """
    if _is_fsdp(m) and not _is_no_shard(m):
        # FULL_SHARD/SHARD_GRAD_OP：需要聚合全参到当前 rank 显存（offload_to_cpu=False）
        # 注意：rank0_only=True 时其它 rank 不进入 with 体；此处我们自己在外层判断 rank。
        with FSDP.summon_full_params(
            m, writeback=False, rank0_only=True, offload_to_cpu=False, with_grads=False
        ):
            # m.module 是裸模块
            for n, p in m.module.named_parameters():
                yield n, p
    else:
        # NO_SHARD 或 非 FSDP：直接拿参数（m.module 若存在则用其，否则 m 本身）
        inner = getattr(m, "module", m)
        for n, p in inner.named_parameters():
            yield n, p


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999, trainable_only: bool = False):
        """
        Args:
            fsdp_module: 可以是 FSDP 包装模块，也可以是普通 nn.Module
            decay: EMA 衰减系数，越接近 1 越平滑
            trainable_only: 如果为True，只对requires_grad=True的参数做EMA，大幅减少内存和计算开销
        """
        self.decay = decay
        self.trainable_only = trainable_only
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module: torch.nn.Module):
        """
        初始化 EMA shadow 到 CPU（float32）。
        - 所有 rank 都进入 _iter_named_params_full，避免 FSDP deadlock
        - 只有 rank0 实际保存 shadow
        - 如果 trainable_only=True，只保存 requires_grad=True 的参数
        """
        is_rank0 = (not _is_dist_init()) or (dist.get_rank() == 0)

        num_total = 0
        num_trainable = 0
        total_params = 0
        trainable_params = 0

        for n, p in _iter_named_params_full(fsdp_module):
            num_total += 1
            total_params += p.numel()

            # 只对可训练参数做EMA（如果启用trainable_only）
            if self.trainable_only and not p.requires_grad:
                continue

            if p.requires_grad:
                num_trainable += 1
                trainable_params += p.numel()

            if is_rank0:
                self.shadow[n] = p.detach().to(dtype=torch.float32, device="cpu").clone()

        if is_rank0 and self.trainable_only:
            print(f"[EMA_FSDP] trainable_only=True: tracking {num_trainable}/{num_total} parameters "
                  f"({trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M params, "
                  f"{trainable_params/total_params*100:.2f}% of total)")

        if _is_dist_init():
            dist.barrier()

    @torch.no_grad()
    def update(self, fsdp_module: torch.nn.Module):
        """
        用当前模型权重更新 EMA（仅 rank0 执行 EMA 累计）。
        所有 rank 都必须参与 _iter_named_params_full 以避免 FSDP 死锁。
        如果 trainable_only=True，只更新 requires_grad=True 的参数。
        """
        is_rank0 = (not _is_dist_init()) or (dist.get_rank() == 0)

        d = self.decay
        for n, p in _iter_named_params_full(fsdp_module):
            # 只对可训练参数做EMA（如果启用trainable_only）
            if self.trainable_only and not p.requires_grad:
                continue

            # 只有 rank0 实际更新 EMA shadow
            if is_rank0:
                # 将当前权重取到 CPU/float32，再执行 EMA
                cur = p.detach().to(dtype=torch.float32, device="cpu")
                sp = self.shadow.get(n, None)
                if sp is None:
                    self.shadow[n] = cur.clone()
                else:
                    sp.mul_(d).add_(cur, alpha=1.0 - d)

        if _is_dist_init():
            dist.barrier()

    # 仅在保存/评估时由 rank0 使用
    def state_dict(self):
        """
        返回可序列化的 EMA 权重（CPU/float32）。非 rank0 也可调用（返回本地 shadow）。
        """
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        """加载 EMA 权重到 shadow（CPU/float32）。"""
        self.shadow = {k: v.to(dtype=torch.float32, device="cpu").clone() for k, v in sd.items()}

    @torch.no_grad()
    def copy_to(self, fsdp_module: torch.nn.Module):
        """
        将 EMA 权重写入模型参数（仅 rank0 执行）。
        - 对 FSDP(FULL_SHARD) 情况：在 summon_full_params(writeback=True, offload_to_cpu=False) 内部写回，
          退出上下文时 FSDP 会将全参重新分片到各 rank。
        - NO_SHARD/非 FSDP：直接覆盖参数即可。
        所有 rank 都必须参与以避免 FSDP 死锁。
        """
        is_rank0 = (not _is_dist_init()) or (dist.get_rank() == 0)

        if _is_fsdp(fsdp_module) and not _is_no_shard(fsdp_module):
            # FULL_SHARD/SHARD_GRAD_OP：需要 summon_full_params 以便写回
            # Check if model has PEFT LoRA adapters that would cause shape mismatch
            has_lora = hasattr(fsdp_module, 'peft_config') or hasattr(getattr(fsdp_module, 'module', None), 'peft_config')
            writeback_enabled = not has_lora  # Disable writeback for LoRA models

            with FSDP.summon_full_params(
                fsdp_module, writeback=writeback_enabled, rank0_only=True, offload_to_cpu=False, with_grads=False
            ):
                # 只有 rank0 实际写入 EMA 权重
                if is_rank0:
                    inner = fsdp_module.module
                    for n, p in inner.named_parameters():
                        if n in self.shadow:
                            src = self.shadow[n].to(dtype=p.dtype, device=p.device)
                            p.data.copy_(src)
        else:
            # NO_SHARD 或 非 FSDP：直接覆盖
            if is_rank0:
                inner = getattr(fsdp_module, "module", fsdp_module)
                for n, p in inner.named_parameters():
                    if n in self.shadow:
                        src = self.shadow[n].to(dtype=p.dtype, device=p.device)
                        p.data.copy_(src)

        if _is_dist_init():
            dist.barrier()

# === misc ===

import numpy as np
import random
import torch
import wandb
from torchvision.utils import make_grid
import os
import imageio
import sys
import time
from typing import Tuple, List
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

# ============================================================================
# Training Utilities
# ============================================================================



def nymeria_worker_init_fn(worker_id):
    """Worker initialization function for Nymeria dataset with LMDB.

    Reopens LMDB environments in worker processes to avoid sharing across processes.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "reopen_envs"):
        worker_info.dataset.reopen_envs()


def find_safe_lora_targets(model: nn.Module, allow_conv1x1: bool = False) -> list[str]:
    """Find all safe LoRA injection targets in a model.

    Returns末级模块名 (last-level module names) that are suitable for LoRA:
    - Always includes: nn.Linear
    - Optional: nn.Conv1d/2d/3d with kernel_size=1 (pseudo-linear projection)

    Note: PEFT uses substring/equality matching on last-level names, so ensure
    names don't collide with activations/normalization layers.

    Args:
        model: PyTorch model to search
        allow_conv1x1: Whether to include 1x1 convolutions

    Returns:
        Sorted list of unique module names suitable for LoRA
    """
    names = set()

    for full_name, m in model.named_modules():
        last = full_name.rsplit('.', 1)[-1]

        if isinstance(m, nn.Linear):
            names.add(last)
            continue

        if allow_conv1x1 and isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
            if all(k == 1 for k in ks):
                names.add(last)

    return sorted(names)


# ============================================================================
# Random Seed Setting
# ============================================================================

def set_seed(seed: int = 42):
    """
    设置随机种子，保证结果可复现。
    同一个代码在不同机器/进程上输出一致。
    """
    random.seed(seed)  # Python 自带随机
    np.random.seed(seed)  # NumPy 随机
    torch.manual_seed(seed)  # CPU 上的 Torch
    torch.cuda.manual_seed(seed)  # GPU 上的 Torch
    torch.cuda.manual_seed_all(seed)  # 多卡一致

    # 确保 cudnn 可复现（可能牺牲速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 确保 hash 随机性固定（Python 3.2+）
    os.environ["PYTHONHASHSEED"] = str(seed)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict

def prepare_for_saving(tensor, fps=16, caption=None):
    # Convert range [-1, 1] to [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()
    if tensor.ndim == 4:
        # Assuming it's an image and has shape [batch_size, 3, height, width]
        tensor = make_grid(tensor, 4, padding=0, normalize=False)
        return wandb.Image((tensor * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8), caption=caption)
    elif tensor.ndim == 5:
        # Assuming it's a video and has shape [batch_size, num_frames, 3, height, width]
        return wandb.Video((tensor * 255).cpu().numpy().astype(np.uint8), fps=fps, format="mp4", caption=caption)
    else:
        raise ValueError("Unsupported tensor shape for saving. Expected 4D (image) or 5D (video) tensor.")

import random

def apply_prompt_dropout(prompts, dropout_prob=0.2, placeholder=""):
    """Apply dropout to a list of prompts with given probability"""
    return [
        prompt if random.random() > dropout_prob else placeholder
        for prompt in prompts
    ]


import os
import torch
import imageio
import numpy as np
from torchvision.utils import save_image




import gc
import inspect
import os
import shutil
import subprocess
import time

import cv2
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs




# ============================================================================
# Color / Camera Helper Functions
# ============================================================================

def color_transfer(sc, dc):
    """
    Transfer color distribution from sc to dc (per-channel mean/std in RGB).
    sc/dc are uint8 RGB images.
    """
    def get_mean_and_std(img):
        mean, std = cv2.meanStdDev(img)
        mean = mean.flatten()
        std = std.flatten()
        return mean, std

    sc = sc.astype(np.float32)
    dc = dc.astype(np.float32)

    sc_mean, sc_std = get_mean_and_std(sc)
    dc_mean, dc_std = get_mean_and_std(dc)

    sc_std = np.where(sc_std < 1e-6, 1.0, sc_std)
    result = (sc - sc_mean) * (dc_std / sc_std) + dc_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def custom_meshgrid(*args):
    """torch.meshgrid with backward-compatible indexing."""
    if "indexing" in torch.meshgrid.__code__.co_varnames:
        return torch.meshgrid(*args, indexing="ij")
    return torch.meshgrid(*args)


def get_relative_pose(cam_params):
    """Return camera-to-world poses relative to the first frame."""
    if len(cam_params) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)
    poses = [cam.c2w_mat for cam in cam_params]
    ref = poses[0]
    ref_inv = np.linalg.inv(ref)
    rel = [ref_inv @ p for p in poses]
    return np.stack(rel, axis=0).astype(np.float32)


def ray_condition(K, c2ws, height, width):
    """
    Build Plücker embeddings [B, N, H, W, 6] from intrinsics/extrinsics.
    K: [B, N, 4] (fx, fy, cx, cy); c2ws: [B, N, 4, 4].
    """
    device = c2ws.device
    dtype = c2ws.dtype

    fx = K[..., 0].unsqueeze(-1).unsqueeze(-1)
    fy = K[..., 1].unsqueeze(-1).unsqueeze(-1)
    cx = K[..., 2].unsqueeze(-1).unsqueeze(-1)
    cy = K[..., 3].unsqueeze(-1).unsqueeze(-1)

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = custom_meshgrid(ys, xs)
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)

    dirs_cam = torch.stack([(xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)], dim=-1)
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True).clamp(min=1e-8)

    rot = c2ws[..., :3, :3]
    trans = c2ws[..., :3, 3]
    dirs_world = torch.matmul(rot.unsqueeze(-3).unsqueeze(-3), dirs_cam.unsqueeze(-1)).squeeze(-1)

    origins = trans.unsqueeze(-2).unsqueeze(-2).expand_as(dirs_world)
    moments = torch.cross(origins, dirs_world, dim=-1)
    plucker = torch.cat([dirs_world, moments], dim=-1)
    return plucker


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)


def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image




def timer(func):
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        end_time    = time.time()
        print(f"function {func.__name__} running for {end_time - start_time} seconds")
        return result
    return wrapper


def _write_to_excel(model_name, time_sum):
    row_env = os.environ.get(f"{model_name}_EXCEL_ROW", "1")  # 默认第1行
    col_env = os.environ.get(f"{model_name}_EXCEL_COL", "1")  # 默认第A列
    file_path = os.environ.get("EXCEL_FILE", "timing_records.xlsx")  # 默认文件名

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
    except FileNotFoundError:
        df = pd.DataFrame()

    row_idx = int(row_env)
    col_idx = int(col_env)

    if row_idx >= len(df):
        df = pd.concat([df, pd.DataFrame([ [None] * (len(df.columns) if not df.empty else 0) ] * (row_idx - len(df) + 1))], ignore_index=True)

    if col_idx >= len(df.columns):
        df = pd.concat([df, pd.DataFrame(columns=range(len(df.columns), col_idx + 1))], axis=1)

    df.iloc[row_idx, col_idx] = time_sum

    df.to_excel(file_path, index=False, header=False, sheet_name="Sheet1")



# ============================================================================
# Camera Control Helper Functions (from VideoX-Fun)
# ============================================================================
class Camera(object):
    """Camera class for processing camera parameters.
    Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

        # #! dzc
        # self.c2w_mat = fix_c2w(self.c2w_mat)








def ensure_pathlib_local_stub() -> None:
    module_name = "pathlib._local"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)
    stub.Path = pathlib.Path
    stub.PosixPath = getattr(pathlib, "PosixPath", pathlib.Path)
    stub.WindowsPath = getattr(pathlib, "WindowsPath", pathlib.Path)
    sys.modules[module_name] = stub



def process_pose_file(camera_poses_text, width=512, height=512, original_pose_width=512, original_pose_height=512, return_poses=False):
    """Process camera pose file and convert to Plucker embedding for camera control.
    Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py

    Args:
        pose_file_path: Path to the pose txt file
        width: Target width
        height: Target height
        original_pose_width: Original pose width
        original_pose_height: Original pose height
        return_poses: If True, return raw poses instead of Plucker embedding

    Returns:
        Plucker embedding [num_frames, height, width, 6] or raw poses
    """
    poses = [pose.strip().split(' ') for pose in camera_poses_text[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width # dzc: 
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height
        
        #! 反归一化，将intrinsic调整至target图像大小
        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)
        K = torch.as_tensor(intrinsic)[None]  # [1, num_frames, 4]
        c2ws = get_relative_pose(cam_params)
        c2ws = torch.as_tensor(c2ws)[None]  # [1, num_frames, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width)[0].permute(0, 3, 1, 2).contiguous()  # num_frames, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding


def process_camera_control_latent(
    control_camera_video: torch.Tensor,
) -> torch.Tensor:
    """
    Process camera control video (Plucker embeddings) into control_camera_latents_input.

    This follows the logic from WanFunControlPipeline:
    1. Repeat first frame 4 times and concatenate with remaining frames
    2. Transpose dimensions
    3. Reshape and reorganize into blocks of 4 frames

    Args:
        control_camera_video: Plucker embedding [num_frames, H, W, 6]
        num_frames: Number of output frames
        height: Video height
        width: Video width

    Returns:
        control_camera_latents_input: Processed tensor [1, 24, num_latent_frames, H/8, W/8]
    """
    # Convert [num_frames, H, W, 6] to [1, 6, num_frames, H, W]
    control_camera_video = control_camera_video.permute(3, 0, 1, 2).unsqueeze(0)

    # Rearrange dimensions: Concatenate repeated first frame with rest
    # torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2) -> [1, 6, 4, H, W]
    # control_camera_video[:, :, 1:] -> [1, 6, num_frames-1, H, W]
    control_camera_latents = torch.concat(
        [
            torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
            control_camera_video[:, :, 1:]
        ], dim=2
    ).transpose(1, 2)  # [1, num_frames+3, 6, H, W]

    # Reshape into blocks of 4: view as groups of 4 frames
    b, f, c, h, w = control_camera_latents.shape
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    # Now: [1, (num_frames+3)//4, 6, 4, H, W]

    # Flatten the 6x4 into 24 channels
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
    # Final: [1, 24, (num_frames+3)//4, H, W]

    return control_camera_latents


# ============================================================================
# CFG Optimization (moved from cfg_optimization.py)
# ============================================================================

def cfg_skip():
    def decorator(func):
        def wrapper(self, x, *args, **kwargs):
            bs = len(x)
            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                bs_half = int(bs // 2)

                new_x = x[bs_half:]

                new_args = []
                for arg in args:
                    if isinstance(arg, (torch.Tensor, list, tuple, np.ndarray)):
                        new_args.append(arg[bs_half:])
                    else:
                        new_args.append(arg)

                new_kwargs = {}
                for key, content in kwargs.items():
                    if isinstance(content, (torch.Tensor, list, tuple, np.ndarray)):
                        new_kwargs[key] = content[bs_half:]
                    else:
                        new_kwargs[key] = content
            else:
                new_x = x
                new_args = args
                new_kwargs = kwargs

            result = func(self, new_x, *new_args, **new_kwargs)

            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                result = torch.cat([result, result], dim=0)

            return result
        return wrapper
    return decorator


# ============================================================================
# Task Utilities (moved from task_utils.py)
# ============================================================================

def crop_latent_frames(target_latent: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Crop latent frames to match the specified number of frames.

    Args:
        target_latent: Target video latents [B, F, C, H, W]
        num_frames: Desired number of frames

    Returns:
        Cropped latent tensor

    Raises:
        ValueError: If data has fewer frames than required
    """
    data_num_frames = target_latent.shape[1]

    if data_num_frames > num_frames:
        target_latent = target_latent[:, :num_frames, :, :, :]
    elif data_num_frames < num_frames:
        raise ValueError(
            f"Data has {data_num_frames} latent frames but config requires {num_frames} frames."
        )
    return target_latent


def process_timestep(
    timestep: torch.Tensor,
    task_type: str,
    num_frame_per_block: int = None
) -> torch.Tensor:
    """
    Pre-process timestep based on generator task type.

    Args:
        timestep: [batch_size, num_frame] tensor
        task_type: "image", "bidirectional_video", or "causal_video"
        num_frame_per_block: Number of frames per block (required for causal_video)

    Returns:
        Processed timestep tensor
    """
    if task_type == "image":
        assert timestep.shape[1] == 1
        return timestep
    elif task_type == "bidirectional_video":
        for index in range(timestep.shape[0]):
            timestep[index] = timestep[index, 0]
        return timestep
    elif task_type == "causal_video":
        if num_frame_per_block is None:
            raise ValueError("num_frame_per_block is required for causal_video task type")
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep
    else:
        raise NotImplementedError(f"Unsupported model type {task_type}")


def get_timestep(
    min_timestep: int,
    max_timestep: int,
    batch_size: int,
    num_frame: int,
    device: torch.device,
    num_frame_per_block: int = None,
    uniform_timestep: bool = False
) -> torch.Tensor:
    """
    Randomly generate timestep tensor.

    Args:
        min_timestep: Minimum timestep value
        max_timestep: Maximum timestep value
        batch_size: Batch size
        num_frame: Number of frames
        device: Device to create tensor on
        num_frame_per_block: Frames per block (required if not uniform)
        uniform_timestep: If True, use same timestep for all frames

    Returns:
        Timestep tensor [batch_size, num_frame]
    """
    if uniform_timestep:
        timestep = torch.randint(
            min_timestep, max_timestep,
            [batch_size, 1],
            device=device, dtype=torch.long
        ).repeat(1, num_frame)
    else:
        if num_frame_per_block is None:
            raise ValueError("num_frame_per_block is required when uniform_timestep=False")
        timestep = torch.randint(
            min_timestep, max_timestep,
            [batch_size, num_frame],
            device=device, dtype=torch.long
        )
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
    return timestep


def diffusion_forcing(
    image_or_video_shape: Tuple[int, ...],
    target_latent: torch.Tensor,
    denoising_step_list: torch.Tensor,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    generator_task: str,
    num_frame_per_block: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate noisy input using diffusion forcing.

    Args:
        image_or_video_shape: Shape tuple (B, F, C, H, W)
        target_latent: Target latent tensor
        denoising_step_list: List of denoising steps
        scheduler: Noise scheduler
        device: Device
        dtype: Data type
        generator_task: Task type for timestep processing
        num_frame_per_block: Frames per block (for causal tasks)

    Returns:
        Tuple of (noisy_input, noise, timestep)
    """
    simulated_noisy_input = []
    simulated_noises = []
    B, F, C, H, W = image_or_video_shape

    for timestep in denoising_step_list:
        noise = torch.randn(image_or_video_shape, device=device, dtype=dtype)
        noisy_timestep = timestep * torch.ones(image_or_video_shape[:2], device=device, dtype=torch.long)

        if timestep != 0:
            noisy_image = scheduler.add_noise(
                target_latent.flatten(0, 1),
                noise.flatten(0, 1),
                noisy_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])
        else:
            noisy_image = target_latent

        simulated_noisy_input.append(noisy_image)
        simulated_noises.append(noise)

    simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)
    simulated_noises = torch.stack(simulated_noises, dim=1)

    index = torch.randint(
        0, len(denoising_step_list),
        [image_or_video_shape[0], image_or_video_shape[1]],
        device=device, dtype=torch.long
    )
    index = process_timestep(index, task_type=generator_task, num_frame_per_block=num_frame_per_block)

    noisy_input = torch.gather(
        simulated_noisy_input, dim=1,
        index=index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(-1, -1, -1, *image_or_video_shape[2:])
    ).squeeze(1)

    noise = torch.gather(
        simulated_noises, dim=1,
        index=index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(-1, -1, -1, *image_or_video_shape[2:])
    ).squeeze(1)

    timestep = denoising_step_list[index]
    return noisy_input, noise, timestep


def get_full_state_dict(module) -> dict:
    """
    Get full state dict from FSDP or regular module.

    Args:
        module: PyTorch module (may be wrapped with FSDP)

    Returns:
        State dictionary
    """
    if isinstance(module, FSDP):
        state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, state_dict_cfg):
            state_dict = module.state_dict()
    else:
        state_dict = module.state_dict()
    return state_dict


def create_eval_generator(
    generator_state_dict: dict,
    config,
    device: torch.device,
    dtype: torch.dtype,
    generator_class,
    has_lora: bool,
    step: int = 0
):
    """
    Create evaluation generator with loaded weights.

    Args:
        generator_state_dict: State dict to load
        config: Configuration object
        device: Device to move model to
        dtype: Data type
        generator_class: Class to instantiate (e.g., BidirectionalWanWrapper)
        has_lora: Whether the model uses LoRA
        step: Current training step

    Returns:
        Evaluation generator model
    """
    generator_wrapper = generator_class(config=config, role="generator")

    if has_lora:
        lora_config = LoraConfig(**config.lora_config)
        generator_wrapper = get_peft_model(generator_wrapper, lora_config)
        lora_state_dict = get_peft_model_state_dict(
            generator_wrapper,
            state_dict=generator_state_dict
        )
        set_peft_model_state_dict(generator_wrapper, lora_state_dict)
    else:
        # Always load state_dict if available (including step=0 for initial validation)
        if generator_state_dict:
            generator_wrapper.load_state_dict(generator_state_dict, strict=False)

    generator_wrapper = generator_wrapper.to(device=device, dtype=dtype)
    generator_wrapper.eval()
    return generator_wrapper
