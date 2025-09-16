python src/inference_difix_v1.py \
    --model_path "outputs/difix_without_ref/train/checkpoints/model_224001.pkl" \
    --input_image "assets/00230.png" \
    --prompt "remove degradation" \
    --output_dir "outputs/with_ckpt" \
    --timestep 199
