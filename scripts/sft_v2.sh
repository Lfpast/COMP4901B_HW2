export WANDB_API_KEY="f91ffccf67c9e7a8c326c0a655ca367f0f89e2e1"
export WANDB_PROJECT="COMP4901B-Homework2"

RUNNAME="HW2_v5_ultrabatch"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.135B"
OUTPUTPATH="ckpt"
DEVICES="0"
NUM_GPUS=1

# V5 Ultra Batch: Mimicking your classmate's setup
# Key: VERY LARGE batch size for fast, smooth convergence
TOTALBSZ=512          # 🚀 4x larger than before!
BSZPERDEV=2           # Keep small to fit in memory
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))  # Will be 256!

export CUDA_VISIBLE_DEVICES=${DEVICES}

echo "=========================================="
echo "Training Configuration - HW2_v5_ultrabatch"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps (HUGE!)"
echo "Effective batch size: ${TOTALBSZ} (4x normal)"
echo "Learning rate: 8e-5 (scaled with batch size)"
echo "Warmup ratio: 0.4 (40% warmup for stability)"
echo "Max grad norm: 1.0 (gradient clipping)"
echo "Epochs: 2 (reduced for fast convergence)"
echo "Sequence length: 2048"
echo ""
echo "Expected training steps: ~24 (6000÷512×2)"
echo "=========================================="
echo ""
echo "V5 Strategy (Classmate's Secret):"
echo "🔥 Ultra-large batch (512): Fewer, more stable updates"
echo "🔥 Scaled learning rate (8e-5): Matches large batch"
echo "🔥 Strong warmup (40%): Prevents early instability"
echo "🔥 Gradient clipping (1.0): Ultra-smooth curve"
echo "🔥 Fewer epochs (2): Fast finish in <30 steps"
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
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 8e-5 \
    --warmup_ratio 0.4 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --dataloader_num_workers 4 \
    --preprocess_workers 4 \
    --max_rounds 5

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoint saved to: ${OUTPUTPATH}/${RUNNAME}"
echo ""
echo "Expected results:"
echo "✅ Total steps: ~24 (similar to classmate's <100)"
echo "✅ Smooth curve: Gradient clipping + large batch"
echo "✅ Fast convergence: High LR + strong warmup"
echo "✅ Final loss: Should reach ~1.0-1.05"
echo ""
echo "Next: Run evaluation and compare with classmate!"
echo "=========================================="
