export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

accelerate launch --mixed_precision=bf16 src/train_difix_mv.py \
    --output_dir=./outputs/difix/train \
    --dataset_path="/mnt/HDD3/essen900718/Difix3D/data/data_mv.json" \
    --max_train_steps 10000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --checkpointing_steps=5000 --eval_freq 1000 --viz_freq 100 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix_with_ref" --tracker_run_name "train" --timestep 199 \
    --mv_unet
