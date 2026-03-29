#!/bin/bash
# LiveWorld Training
export TORCH_HOME=./ckpts
export HF_HUB_DISABLE_IMPORTS_VERIFICATION=1
export NCCL_TIMEOUT=7200

# === Define nodes and per-node GPU assignments ===
NODES=("$(hostname)")
CUDA_VISIBLE_DEVICES_LIST=("1")

# === Auto-detect current node ===
HOSTNAME=$(hostname)
for i in "${!NODES[@]}"; do
    if [[ "$HOSTNAME" == "${NODES[$i]}" ]]; then
        NODE_RANK=$i
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]}
        break
    fi
done

if [[ -z "$NODE_RANK" ]]; then
    echo "Error: hostname ($HOSTNAME) not in NODES list."
    exit 1
fi

MASTER_ADDR=${NODES[0]}
NNODES=${#NODES[@]}
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Node: $HOSTNAME | rank=$NODE_RANK | GPUs=$CUDA_VISIBLE_DEVICES | master=$MASTER_ADDR | nnodes=$NNODES"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29511 \
    scripts/train.py \
    --config_path configs/train_liveworld_1-3B.yaml \
    --disable_wandb
