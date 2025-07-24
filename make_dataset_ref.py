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
  
    valid_pairs = []
    for render_path in render_paths:
        filename = os.path.basename(render_path)
        gt_path = os.path.join(gt_dir, filename)
        if os.path.exists(gt_path):
            valid_pairs.append((filename, render_path, gt_path))
        else:
            print(f"⚠️ Warning: missing target image for {render_path}")

    # add the reference image
    for i, (filename, render_path, gt_path) in enumerate(valid_pairs):
        prev_idx = (i - 1) % len(valid_pairs)
        ref_filename = valid_pairs[prev_idx][0]
        ref_gt_path = os.path.join(gt_dir, ref_filename)

        data_id = f"{folder}_{os.path.splitext(filename)[0]}"
        sample = {
            "id": data_id,
            "image": render_path,
            "target_image": gt_path,
            "ref_image": ref_gt_path,
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
for split_name, split_samples in [("train", train_samples), ("test", test_samples)]:
    for sample in split_samples:
        output_json[split_name][sample["id"]] = {
            "image": sample["image"],
            "target_image": sample["target_image"],
            "ref_image": sample["ref_image"],
            "prompt": sample["prompt"]
        }

with open(os.path.join(base_dir, "data.json"), "w") as f:
    json.dump(output_json, f, indent=4)

print("✅ Saved data.json with ref_image included.")
