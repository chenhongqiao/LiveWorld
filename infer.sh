#!/bin/bash
# LiveWorld Inference
export CUDA_VISIBLE_DEVICES=0

python scripts/infer.py \
    --config examples/inference_sample/processed/kid_coffee/infer_scripts/case1_right.yaml \
    --system-config configs/infer_system_config_few_step_14B.yaml \
    --output-root outputs \
    --device cuda:0
