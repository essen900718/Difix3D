python src/inference_difix_v1.py \
    --model_path "outputs/difix_with_ref/train/checkpoints/model_229001.pkl" \
    --input_image "assets/00230.png" \
    --ref_image "/mnt/HDD2/essen900718/Difix3D/assets/ref_image/00003.png" \
    --prompt "remove degradation" \
    --output_dir "outputs/with_ckpt_with_ref" \
    --timestep 199
