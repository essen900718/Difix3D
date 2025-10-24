import json
import torch
from PIL import Image
import torchvision.transforms.functional as F


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=640, width=384, tokenizer=None):
        print("MV Dataset Loaded")
        # [x] height=576, width=1024 is the original size of difix
        # [v] height=800, width=544 is the size that suitable for the trainiing data and without reference image 
        # [x] height=768, width=512 is the size that suitable for the training data and with reference image (still OOM)
        # [v] height=640, width=384 is the size that suitable for the training data and with reference image
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.img_ids)

    def load_and_preprocess(self, path):
        """Load image and normalize to [-1,1] range"""
        img = Image.open(path).convert("RGB")
        img_t = F.to_tensor(img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        item = self.data[img_id]

        input_imgs = item["image"]
        output_imgs = item["target_image"]
        captions = item["prompt"]

        # 確保都是 list（有些資料可能是單張）
        if isinstance(input_imgs, str):
            input_imgs = [input_imgs]
        if isinstance(output_imgs, str):
            output_imgs = [output_imgs]
        if isinstance(captions, str):
            captions = [captions]

        try:
            # 處理多張圖片
            input_tensors = [self.load_and_preprocess(p) for p in input_imgs]
            output_tensors = [self.load_and_preprocess(p) for p in output_imgs]
        except Exception as e:
            print(f"Error loading image(s) for {img_id}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # [V, C, H, W]
        input_tensors = torch.stack(input_tensors, dim=0)
        output_tensors = torch.stack(output_tensors, dim=0)

        out = {
            "conditioning_pixel_values": input_tensors,  # (V, C, H, W)
            "output_pixel_values": output_tensors,       # (V, C, H, W)
            "caption": captions,                         # list of captions
        }

        # Tokenize caption list if tokenizer exists
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                captions,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            out["input_ids"] = tokenized.input_ids  # (V, seq_len)

        return out
