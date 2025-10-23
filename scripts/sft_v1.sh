#!/bin/bash
# Optimized SFT configuration for 2080Ti (11GB VRAM)
# Target: Achieve strict accuracy significantly higher than 22% (25-35 points)
#
# Key optimizations:
# 1. Higher learning rate (3e-5) for faster convergence
# 2. More epochs (5) to ensure thorough learning
# 3. Longer warmup (0.15) for stability
# 4. Cosine decay with restarts for better optimization
# 5. Optimized batch size and sequence length for 11GB VRAM

export WANDB_API_KEY=""
export WANDB_PROJECT="COMP4901B-Homework2"

RUNNAME="HW2_v1"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="135M"
OUTPUTPATH="ckpt"
DEVICES="0"
NUM_GPUS=1

# Optimized batch configuration for 2080Ti 11GB
TOTALBSZ=128          # Keep effective batch size for stable gradients
BSZPERDEV=2           # Increased from 1 to 2 (11GB can handle this)
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))

export CUDA_VISIBLE_DEVICES=${DEVICES}

echo "=========================================="
echo "Training Configuration - HW2_v1"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps"
echo "Effective batch size: ${TOTALBSZ}"
echo "Learning rate: 3e-5 (higher for faster learning)"
echo "Epochs: 5 (more epochs for better convergence)"
echo "Sequence length: 1536 (optimized for 11GB VRAM)"
echo "=========================================="

python train_hw_parallel.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 100 \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --model_max_length 1536 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --flash_attn False \
    --dataloader_num_workers 4 \
    --preprocess_workers 4 \
    --max_rounds 5 \
    --gradient_checkpointing True \
    --optim "adamw_torch"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoint saved to: ${OUTPUTPATH}/${RUNNAME}"
echo ""
echo "Next steps:"
echo "1. Run evaluation:"
echo "   cd ifeval"
echo "   python run_ifeval_local.py --mode all --model_path ../${OUTPUTPATH}/${RUNNAME} --output_dir ./results/${RUNNAME}"
echo ""
echo "2. Check strict accuracy in results/${RUNNAME}/summary.json"
echo "   Target: > 22% (baseline: 22%)"
echo "=========================================="
