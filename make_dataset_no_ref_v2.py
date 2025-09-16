import os
import json
import random
from glob import glob
from tqdm import tqdm

base_dir = "data"
folders = [
    "017-v2",
    "024-v2",
    "030-v2",
    "031-v2",
    "032-v2",
    "033-v2",
    "035-v2",
    "036-v2"
]

prompt = "remove degradation"
test_ratio = 0.01
num_cameras = 16

# 固定規則
camera_quota = {i: 400 for i in range(num_cameras)}
for i in [6, 7, 8, 9, 10, 11]:
    camera_quota[i] = 200

all_data = []

for folder in folders:
    full_folder_path = os.path.join(base_dir, folder)
    render_dir = os.path.join(full_folder_path, "renders")
    gt_dir = os.path.join(full_folder_path, "gt")

    render_paths = sorted(glob(os.path.join(render_dir, "*.png")))
    print(f"Processing folder: {folder}, found {len(render_paths)} render images")

    # 根據 camera index 分組
    camera_dict = {i: [] for i in range(num_cameras)}
    for render_path in render_paths:
        filename = os.path.basename(render_path)
        frame_id = int(os.path.splitext(filename)[0])
        cam_idx = frame_id % num_cameras
        gt_path = os.path.join(gt_dir, filename)
        if os.path.exists(gt_path):
            camera_dict[cam_idx].append((render_path, gt_path))

    folder_samples = []
    for cam_idx, pairs in camera_dict.items():
        k = min(camera_quota[cam_idx], len(pairs)) 
        chosen = random.sample(pairs, k)
        for render_path, gt_path in chosen:
            data_id = f"{folder}_{os.path.splitext(os.path.basename(render_path))[0]}"
            sample = {
                "id": data_id,
                "image": render_path,
                "target_image": gt_path,
                "prompt": prompt
            }
            folder_samples.append(sample)

    print(f"Selected {len(folder_samples)} samples for {folder} (expected {sum(camera_quota.values())})")
    all_data.extend(folder_samples)

# # 打亂 & split
# random.seed(42)
# random.shuffle(all_data)
split_idx = int(len(all_data) * (1 - test_ratio))
train_samples = all_data[:split_idx]
test_samples = all_data[split_idx:]

print(f"Total samples: {len(all_data)}, Train: {len(train_samples)}, Test: {len(test_samples)}")

# 輸出 json
output_json = {"train": {}, "test": {}}
for sample in train_samples:
    output_json["train"][sample["id"]] = {
        "image": sample["image"],
        "target_image": sample["target_image"],
        "prompt": sample["prompt"]
    }
for sample in test_samples:
    output_json["test"][sample["id"]] = {
        "image": sample["image"],
        "target_image": sample["target_image"],
        "prompt": sample["prompt"]
    }

with open(os.path.join(base_dir, "data.json"), "w") as f:
    json.dump(output_json, f, indent=4)

print("✅ Saved data.json")
