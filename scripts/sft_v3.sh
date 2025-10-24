#!/bin/bash
# Conservative SFT configuration for 2080Ti (11GB VRAM) - Version 3
# Strategy: Stable training with proven hyperparameters
# Target: Reliable > 22% strict accuracy with minimal risk of instability
#
# Key features:
# 1. Moderate learning rate (2.5e-5) - balanced approach
# 2. Standard epochs (4) - enough for convergence
# 3. Extended warmup (0.2) - maximum stability
# 4. Full sequence length (2048) - maximum context

export WANDB_API_KEY="f91ffccf67c9e7a8c326c0a655ca367f0f89e2e1"
export WANDB_PROJECT="COMP4901B-Homework2"

RUNNAME="HW2_v3"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="135M"
OUTPUTPATH="ckpt"
DEVICES="0"
NUM_GPUS=1

# Conservative batch configuration
TOTALBSZ=128          # Effective batch size
BSZPERDEV=1           # Conservative for full 2048 length
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))

export CUDA_VISIBLE_DEVICES=${DEVICES}

echo "=========================================="
echo "Training Configuration - HW2_v3 (CONSERVATIVE)"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps"
echo "Effective batch size: ${TOTALBSZ}"
echo "Learning rate: 2.5e-5 (moderate)"
echo "Epochs: 4 (balanced)"
echo "Sequence length: 2048 (full context)"
echo "Warmup: 0.2 (extended for stability)"
echo "=========================================="

python train_hw_parallel.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 4 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 100 \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2.5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --flash_attn False \
    --dataloader_num_workers 2 \
    --preprocess_workers 2 \
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
echo "   Target: > 22% (stable baseline)"
echo "=========================================="
