#!/bin/bash
# Download all required pretrained weights into ckpts/
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "Downloading LiveWorld model weights"
echo "============================================================"

# LiveWorld weights (State Adapter + LoRA)
echo ">>> LiveWorld State Adapter + LoRA"
huggingface-cli download ZichengD/LiveWorld ckpts/state_adapter/model.pt --local-dir .
huggingface-cli download ZichengD/LiveWorld ckpts/lora/model.pt --local-dir .

# Wan2.1 T2V 14B backbone
echo ">>> Wan2.1 T2V 14B"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ckpts/Wan-AI--Wan2.1-T2V-14B

# Wan2.1 VAE (for data preparation)
echo ">>> Wan2.1 VAE"
huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir ckpts/alibaba-pai--Wan2.1-Fun-1.3B-InP

# Wan2.1 distilled backbone (for fast inference)
echo ">>> Wan2.1 Distilled Backbone"
huggingface-cli download lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill --local-dir ckpts/Wan2.1-T2V-14B-StepDistill

# Qwen3-VL 8B (entity detection)
echo ">>> Qwen3-VL 8B"
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ckpts/Qwen--Qwen3-VL-8B-Instruct

# SAM3 (video segmentation)
echo ">>> SAM3"
huggingface-cli download facebook/sam3 --local-dir ckpts/facebook--sam3

# Stream3R (3D reconstruction)
echo ">>> Stream3R"
huggingface-cli download yslan/STream3R --local-dir ckpts/yslan--STream3R

# DINOv3 (entity matching, optional)
echo ">>> DINOv3 (optional)"
huggingface-cli download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir ckpts/facebook--dinov3-vith16plus-pretrain-lvd1689m

echo ""
echo "============================================================"
echo "All weights downloaded to ckpts/"
echo "============================================================"
