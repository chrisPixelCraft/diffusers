export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export INSTANCE_DIR="./calli"
export INSTANCE_DIR="./ziplora_dataset/content"
# export OUTPUT_DIR="lora-trained-xl-calli-2000"
export OUTPUT_DIR="lora-trained-xl-content"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

# Define the content prompt
export PROMPT_CONTENT="a cat wearing wearable glasses"
export VALID_PROMPT_CONTENT="A cat wearing wearable glasses"

# Define the style prompt
export PROMPT_STYLE="a photo of watercolour style"
export VALID_PROMPT_STYLE="A photo of watercolour style"

# Define the ranks to loop over
export RANKS=(4 8 16 32 64)

for RANK in "${RANKS[@]}"; do
  echo "Training with instance_dir: $INSTANCE_DIR, output_dir: ${OUTPUT_DIR}_rank_${RANK}, rank: $RANK"

  accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir="${OUTPUT_DIR}_rank_${RANK}" \
    --mixed_precision="bf16" \
    --rank=$RANK \
    --instance_prompt="${PROMPT_CONTENT}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --report_to="wandb" \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --validation_prompt="${VALID_PROMPT_CONTENT}" \
    --validation_epochs=100 \
    --seed="0"
    # --push_to_hub
done

# Second combination: instance_dir = ./autumn, output_dir = lora-trained-xl-autumn
export INSTANCE_DIR="./ziplora_dataset/style"
export OUTPUT_DIR="lora-trained-xl-style"

for RANK in "${RANKS[@]}"; do
  echo "Training with instance_dir: $INSTANCE_DIR, output_dir: ${OUTPUT_DIR}_rank_${RANK}, rank: $RANK"

  accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir="${OUTPUT_DIR}_rank_${RANK}" \
    --mixed_precision="bf16" \
    --rank=$RANK \
    --instance_prompt="${PROMPT_STYLE}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --report_to="wandb" \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --validation_prompt="${VALID_PROMPT_STYLE}" \
    --validation_epochs=100 \
    --seed="0"
    # --push_to_hub
done