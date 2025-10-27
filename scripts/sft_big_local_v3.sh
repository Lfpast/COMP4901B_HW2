export WANDB_API_KEY="f91ffccf67c9e7a8c326c0a655ca367f0f89e2e1"

export WANDB_PROJECT="COMP4901B-Homework2"
RUNNAME="HW2_big_local_v3"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.6B"
OUTPUTPATH="ckpt"
DEVICES="0"  # e.g. 0,1,2,3
NUM_GPUS=1
TOTALBSZ=512
BSZPERDEV=1

LR=6e-5
SEQLENGTH=2048
EPOCH=5
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))
export CUDA_VISIBLE_DEVICES=${DEVICES}
echo "=========================================="
echo "Training Configuration - HW2"
echo "=========================================="
echo "Model: ${MODELPATH} (${MODEL_SIZE})"
echo "Dataset: ${DATAPATH}"
echo "GPUs: ${NUM_GPUS} (Device ${DEVICES})"
echo "Batch size per device: ${BSZPERDEV}"
echo "Gradient accumulation: ${GRADACC} steps"
echo "Effective batch size: ${TOTALBSZ}"
echo "Learning rate: ${LR}"
echo "Epochs: ${EPOCH}"
echo "Sequence length: ${SEQLENGTH}"
echo "=========================================="

# Single GPU training without DeepSpeed
# Using AdamW 8-bit optimizer for large batch training to save memory
# Adding gradient clipping for training stability
python train_hw_parallel.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --learning_rate ${LR} \
    --warmup_ratio 0.3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --model_max_length  ${SEQLENGTH}\
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --flash_attn False \
    --dataloader_num_workers 4 \
    --preprocess_workers 4 \
    --weight_decay 0.01 \
    --max_rounds 6 \
    --optim "adafactor" \
    --max_grad_norm 1.0