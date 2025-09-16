TIMESTEP=149
IMAGE_NAME="00221"

python src/inference_difix_test.py \
    --model_path "/mnt/HDD2/essen900718/Difix3D/outputs/difix/train/checkpoints/model_10001.pkl" \
    --input_image "assets/${IMAGE_NAME}.png" \
    --prompt "remove degradation" \
    --output_dir "outputs/test" \
    --output_image "${IMAGE_NAME}_t${TIMESTEP}.png" \
    --timestep $TIMESTEP
