export WANDB_API_KEY=""

export WANDB_PROJECT="COMP4901B-Homework2"
RUNNAME="HW2_v1"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.6B"
OUTPUTPATH="ckpt"
DEVICES="3"  # e.g. 0,1,2,3
NUM_GPUS=1

# Optimized configuration for smooth training curve with ~100 steps
# Target: Match classmate's smooth curve (loss 1.65 -> 1.0 in ~100 steps)
TOTALBSZ=256      # Medium-large batch for stability
BSZPERDEV=2       # 2080Ti 11GB can handle this
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))

export CUDA_VISIBLE_DEVICES=${DEVICES}
echo "=========================================="
echo "Training Configuration - Smooth 100 Steps"
echo "=========================================="
echo "Model: ${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps"
echo "Effective batch size: ${TOTALBSZ}"
echo "Expected total steps: ~94 (6000÷256×4)"
echo ""
echo "Key optimizations for smooth curve:"
echo "  - Batch size 256 (2x normal) for stable gradients"
echo "  - Learning rate 5e-5 (2.5x scaled)"
echo "  - 30% warmup (first 28 steps)"
echo "  - Gradient clipping (max_grad_norm=1.0)"
echo "  - Weight decay (0.01) for regularization"
echo "  - 4 epochs to reach ~100 steps"
echo "=========================================="

# Single GPU training without DeepSpeed
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
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.3 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
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
    --max_rounds 5

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Results should show:"
echo "  ✅ Smooth loss curve (minimal oscillation)"
echo "  ✅ Fast convergence (1.65 -> ~1.0)"
echo "  ✅ Total steps ~94 (close to 100)"
echo "  ✅ Model saved at: ${OUTPUTPATH}/${RUNNAME}"
echo ""
echo "Next step: Run IFEval evaluation"
echo "  cd ifeval"
echo "  bash run_transformers.sh ${OUTPUTPATH}/${RUNNAME} ${RUNNAME}"
echo "=========================================="
