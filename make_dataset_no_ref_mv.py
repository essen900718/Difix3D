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

# ---- 合併同一個 folder 的兩筆資料 ----
merged_data = []
for folder in folders:
    folder_samples = [s for s in all_data if s["id"].startswith(folder)]
    random.shuffle(folder_samples)  # 打亂同一人的樣本

    # 兩兩合併
    for i in range(0, len(folder_samples) - 1, 2):
        s1, s2 = folder_samples[i], folder_samples[i + 1]
        merged_id = f"{folder}_{os.path.basename(s1['image']).split('.')[0]}_{os.path.basename(s2['image']).split('.')[0]}"
        merged_sample = {
            "id": merged_id,
            "image": [s1["image"], s2["image"]],
            "target_image": [s1["target_image"], s2["target_image"]],
            "prompt": [s1["prompt"], s2["prompt"]],
        }
        merged_data.append(merged_sample)

print(f"After merging: {len(merged_data)} samples total")

# ---- split ----
split_idx = int(len(merged_data) * (1 - test_ratio))
train_samples = merged_data[:split_idx]
test_samples = merged_data[split_idx:]

print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

# ---- 輸出 json ----
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

with open(os.path.join(base_dir, "data_mv.json"), "w") as f:
    json.dump(output_json, f, indent=4)

print("✅ Saved data.json")
