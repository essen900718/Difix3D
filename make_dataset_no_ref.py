import os
import json
import random
from glob import glob
from tqdm import tqdm

base_dir = "data" 
folders = [
    "{FOLDER1}",
    "{FOLDER2}",
    "{FOLDER3}"
]

prompt = "remove degradation"
test_ratio = 0.01  

all_data = []

for folder in folders:
    full_folder_path = os.path.join(base_dir, folder)
    render_dir = os.path.join(full_folder_path, "renders")
    gt_dir = os.path.join(full_folder_path, "gt")

    render_paths = sorted(glob(os.path.join(render_dir, "*.png")))
    print(f"Processing folder: {folder}, found {len(render_paths)} render images")

    for render_path in tqdm(render_paths, desc=f"Processing {folder}"):
        filename = os.path.basename(render_path)
        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: missing target image for {render_path}")
            continue

        data_id = f"{folder}_{os.path.splitext(filename)[0]}"
        sample = {
            "id": data_id,
            "image": render_path,
            "target_image": gt_path,
            "prompt": prompt
        }
        all_data.append(sample)

random.seed(42)
random.shuffle(all_data)
split_idx = int(len(all_data) * (1 - test_ratio))
train_samples = all_data[:split_idx]
test_samples = all_data[split_idx:]
print(f"Total samples: {len(all_data)}, Train: {len(train_samples)}, Test: {len(test_samples)}")

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

print("✅ Saved training_data.json")
