export WANDB_API_KEY="f91ffccf67c9e7a8c326c0a655ca367f0f89e2e1"
export WANDB_PROJECT="COMP4901B-Homework2"

RUNNAME="HW2_v2"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.6B"
OUTPUTPATH="ckpt"
DEVICES="0"
NUM_GPUS=1

# Aggressive batch configuration
TOTALBSZ=128          # Effective batch size
BSZPERDEV=4           # Higher batch size (possible with shorter sequences)
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))

export CUDA_VISIBLE_DEVICES=${DEVICES}

echo "=========================================="
echo "Training Configuration - HW2_v2 (AGGRESSIVE)"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps"
echo "Effective batch size: ${TOTALBSZ}"
echo "Learning rate: 4e-5 (AGGRESSIVE)"
echo "Epochs: 4 (extended training)"
echo "Sequence length: 1024"
echo "=========================================="

python train_hw_parallel.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 4 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --learning_rate 4e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --model_max_length 1024 \
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
echo "Next steps:"
echo "1. Run evaluation:"
echo "   cd ifeval"
echo "   python run_ifeval_local.py --mode all --model_path ../${OUTPUTPATH}/${RUNNAME} --output_dir ./results/${RUNNAME}"
echo ""
echo "2. Check strict accuracy in results/${RUNNAME}/summary.json"
echo "   Target: > 25% (aggressive goal)"
echo "=========================================="