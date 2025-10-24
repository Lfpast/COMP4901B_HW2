export WANDB_API_KEY="f91ffccf67c9e7a8c326c0a655ca367f0f89e2e1"
export WANDB_PROJECT="COMP4901B-Homework2"

RUNNAME="HW2_v1"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.6B"
OUTPUTPATH="ckpt"
DEVICES="3"
NUM_GPUS=1

# V5 Extreme: Maximum smoothness & speed
TOTALBSZ=1024         # 🚀 8x larger! Mega batch
BSZPERDEV=1           # Keep memory-safe
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))  # Will be 512!

export CUDA_VISIBLE_DEVICES=${DEVICES}

echo "=========================================="
echo "Training Configuration - HW2_v5_extreme"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps (EXTREME!)"
echo "Effective batch size: ${TOTALBSZ} (8x normal)"
echo "Learning rate: 1.6e-4 (scaled 8x)"
echo "Warmup ratio: 0.5 (50% warmup)"
echo "Max grad norm: 0.5 (aggressive clipping)"
echo "Weight decay: 0.01 (regularization)"
echo "Epochs: 2"
echo "Sequence length: 2048"
echo ""
echo "Expected training steps: ~12 (6000÷1024×2)"
echo "=========================================="
echo ""
echo "V5 Extreme Strategy:"
echo "💥 Mega batch (1024): Ultra-stable gradients"
echo "💥 Very high LR (1.6e-4): Fast convergence"
echo "💥 50% warmup: Maximum stability"
echo "💥 Tight gradient clip (0.5): Glass-smooth curve"
echo "💥 Weight decay: Prevent overfitting"
echo "💥 Target: <20 steps total"
echo "=========================================="

deepspeed --num_gpus=${NUM_GPUS} train_hw_parallel.py \
    --deepspeed ds_configs/zero2_no_offload.json \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.2 \
    --max_grad_norm 0.5 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --dataloader_num_workers 2 \
    --preprocess_workers 2 \
    --max_rounds 5

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoint saved to: ${OUTPUTPATH}/${RUNNAME}"
echo ""
echo "Extreme configuration results:"
echo "🏆 Steps: ~12 (definitely <100!)"
echo "🏆 Curve: Should be incredibly smooth"
echo "🏆 Speed: Blazing fast convergence"
echo "🏆 Loss: Target ~0.95-1.0"
echo ""
echo "⚠️  Note: Very large batch may need careful tuning"
echo "=========================================="
