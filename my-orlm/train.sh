set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"   # 
SAVE_PATH="${SAVE_PATH:-./checkpoints/orlm-qwen-mini}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"    # 2048/4096/8192
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1}"


cd "$(dirname "$0")/../ORLM"
python3 -m pip install -r ../my-orlm/requirements.trainonly.txt

python3 -m pip install "peft==0.4.0"

python3 ../my-orlm/prepare_data.py or_instruct_3k.jsonl

mkdir -p "$SAVE_PATH"

python3 -m train.finetune \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --train_dataset_name_or_path or_instruct_3k.jsonl \
  --output_dir "$SAVE_PATH" \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --per_device_eval_batch_size  $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRAD_ACC \
  --evaluation_strategy no \
  --save_strategy no \
  --save_total_limit 1 \
  --preprocessing_num_workers 0 \
  --max_seq_length $MAX_SEQ_LENGTH \
  --learning_rate $LR \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --num_train_epochs $EPOCHS \
  --logging_steps 1 \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --overwrite_output_dir \
  --bf16 False --fp16 True






  python -m ORLM.train.finetune \
  --model_name_or_path Qwen/Qwen3-8b\
  --train_dataset_name_or_path ORLM/or_instruct_3k.jsonl \
  --output_dir ORLM/checkpoints/orlm-qwen-mini \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 8192 \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --fp16 True 2>&1 | tee ORLM/train.log