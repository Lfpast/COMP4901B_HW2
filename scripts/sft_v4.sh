export WANDB_API_KEY="f91ffccf67c9e7a8c326c0a655ca367f0f89e2e1"
export WANDB_PROJECT="COMP4901B-Homework2"

RUNNAME="HW2_v4"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.6B"
OUTPUTPATH="ckpt"
DEVICES="0"
NUM_GPUS=1

# V4 Configuration: Smooth & Stable Training
TOTALBSZ=128          # Effective batch size
BSZPERDEV=1           # Batch size per device (2080Ti can handle this)
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))

export CUDA_VISIBLE_DEVICES=${DEVICES}

echo "=========================================="
echo "Training Configuration - HW2_v4 (Optimized)"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps"
echo "Effective batch size: ${TOTALBSZ}"
echo "Learning rate: 2e-5 (STABLE)"
echo "Warmup ratio: 0.25 (EXTENDED)"
echo "Epochs: 3 (OPTIMAL)"
echo "Sequence length: 2048 (FULL)"
echo "=========================================="
echo ""
echo "V4 Improvements:"
echo "✅ Lower LR (4e-5 → 2e-5) for smoother training"
echo "✅ Extended warmup (0.15 → 0.25) to reduce oscillation"
echo "✅ Reduced epochs (5 → 3) to prevent overfitting"
echo "✅ Full sequence length (1024 → 2048) for better context"
echo "=========================================="

python train_hw_parallel.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.25 \
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
echo "Checkpoint saved to: ${OUTPUTPATH}/${RUNNAME}"
echo ""
echo "Expected improvements:"
echo "✅ Smoother loss curve with less oscillation"
echo "✅ Better convergence to lower loss values"
echo "✅ Improved instruction-following (target: >25% strict accuracy)"
echo ""
echo "Next steps:"
echo "1. Compare loss curves: V3 vs V4 on W&B"
echo "2. Run evaluation:"
echo "   cd ifeval"
echo "   python run_ifeval_transformers.py --model-path ../ckpt/${RUNNAME} --model-name ${RUNNAME}"
echo ""
echo "3. Check strict accuracy in results/${RUNNAME}/summary.json"
echo "=========================================="
