#!/bin/bash
# Fine-tuning script with MPS fallback enabled
# This enables CPU fallback for MPS-unsupported operations

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Default values
CONFIG="${1:-configs/d2city_saliency_enhanced_rtdetr.yml}"
CHECKPOINT="${2:-checkpoints/rtdetr_r101vd_6x_coco.pth}"
AMP_FLAG="${3:---amp}"

echo "Starting fine-tuning with MPS fallback enabled..."
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "AMP: $AMP_FLAG"
echo ""

python3 scripts/finetuning.py \
    --config "$CONFIG" \
    --pretrained-checkpoint "$CHECKPOINT" \
    $AMP_FLAG

