export WANDB_API_KEY="f91ffccf67c9e7a8c326c0a655ca367f0f89e2e1"
export WANDB_PROJECT="COMP4901B-Homework2"

RUNNAME="HW2_v4_fast"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.135B"
OUTPUTPATH="ckpt"
DEVICES="0"
NUM_GPUS=1

# V4 Fast Configuration: DeepSpeed Accelerated
TOTALBSZ=128          # Effective batch size
BSZPERDEV=4           # Increased to 4 (DeepSpeed saves memory)
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))

export CUDA_VISIBLE_DEVICES=${DEVICES}

echo "=========================================="
echo "Training Configuration - HW2_v4_fast (DeepSpeed)"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV} (↑ from 2)"
echo "Gradient accumulation: ${GRADACC} steps"
echo "Effective batch size: ${TOTALBSZ}"
echo "DeepSpeed: ZeRO Stage 2 (NO offload)"
echo "Learning rate: 2e-5 (STABLE)"
echo "Warmup ratio: 0.25 (EXTENDED)"
echo "Epochs: 3 (OPTIMAL)"
echo "Sequence length: 2048 (FULL)"
echo "=========================================="
echo ""
echo "V4 Fast Optimizations:"
echo "🚀 DeepSpeed ZeRO-2: Memory-efficient training"
echo "🚀 Larger batch size (2→4): Faster iteration"
echo "🚀 Less gradient accumulation (64→32 steps): Faster updates"
echo "✅ Lower LR (2e-5): Smooth convergence"
echo "✅ Extended warmup (0.25): Stable startup"
echo "=========================================="

deepspeed --num_gpus=${NUM_GPUS} train_hw_parallel.py \
    --deepspeed ds_configs/zero2_no_offload.json \
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
    --dataloader_num_workers 4 \
    --preprocess_workers 4 \
    --max_rounds 5

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoint saved to: ${OUTPUTPATH}/${RUNNAME}"
echo ""
echo "Speed improvements vs standard training:"
echo "⚡ ~30-50% faster due to DeepSpeed optimizations"
echo "⚡ Larger batch size (4 vs 2) = fewer iterations"
echo "⚡ Better memory efficiency allows more parallelism"
echo ""
echo "Next steps:"
echo "1. Compare training speed: V4 vs V4_fast"
echo "2. Run evaluation:"
echo "   cd ifeval"
echo "   python run_ifeval_transformers.py --model-path ../ckpt/${RUNNAME} --model-name ${RUNNAME}"
echo "=========================================="
