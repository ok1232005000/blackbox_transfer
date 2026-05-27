import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "metadata(2).json"
ENSEMBLE_DIR = ROOT / "ensemble"
OUTPUT_PATH = ROOT / "experiments2" / "records.jsonl"

EPS_VALUES = ("eps4", "eps8", "eps16")
TARGET_MODELS = ("inception_v3", "densenet121", "mobilenet_v2", "efficientnet_b0")


class AdversarialImageDataset(Dataset):
    def __init__(self, image_paths, metadata, transform):
        self.image_paths = image_paths
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_id = image_path.stem
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_id, int(self.metadata[image_id]["true_label"])


def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {str(item["id"]).zfill(4): item for item in metadata}


def build_transform(model_name):
    image_size = 299 if model_name == "inception_v3" else 224
    resize_size = 342 if model_name == "inception_v3" else 256
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model(model_name, device):
    if model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()
    return model


def load_completed_records(output_path):
    completed = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add((str(record["image_id"]).zfill(4), record["surrogate"], record["eps"], record["target_model"]))
    return completed


def get_image_paths(eps, metadata, completed, target_model):
    image_dir = ENSEMBLE_DIR / eps
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    paths = []
    for path in sorted(image_dir.glob("*.png")):
        image_id = path.stem
        key = (image_id, "ensemble", eps, target_model)
        if image_id in metadata and key not in completed:
            paths.append(path)
    return paths


def evaluate_model(model_name, eps, metadata, completed, batch_size, device):
    image_paths = get_image_paths(eps, metadata, completed, model_name)
    if not image_paths:
        print(f"[skip] ensemble/{eps} on {model_name}: already complete")
        return 0

    print(f"[run] ensemble/{eps} on {model_name}: {len(image_paths)} images")
    transform = build_transform(model_name)
    dataset = AdversarialImageDataset(image_paths, metadata, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = build_model(model_name, device)

    written = 0
    with OUTPUT_PATH.open("a", encoding="utf-8") as records_file:
        with torch.inference_mode():
            for images, image_ids, true_labels in loader:
                images = images.to(device, non_blocking=True)
                true_labels = true_labels.to(device, non_blocking=True)
                outputs = model(images)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                predictions = outputs.argmax(dim=1)

                timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                for image_id, true_label, prediction in zip(image_ids, true_labels.cpu().tolist(), predictions.cpu().tolist()):
                    record = {
                        "image_id": str(image_id).zfill(4),
                        "eps": eps,
                        "surrogate": "ensemble",
                        "target_model": model_name,
                        "clean_output": None,
                        "adv_output": int(prediction),
                        "semantic_sim": None,
                        "attack_success": int(prediction) != int(true_label),
                        "timestamp": timestamp,
                    }
                    records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return written


def main():
    parser = argparse.ArgumentParser(description="Run remaining black-box transfer evaluation for ensemble samples.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps", nargs="*", default=list(EPS_VALUES), choices=EPS_VALUES)
    parser.add_argument("--models", nargs="*", default=list(TARGET_MODELS), choices=TARGET_MODELS)
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    completed = load_completed_records(OUTPUT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_written = 0
    for eps in args.eps:
        for model_name in args.models:
            total_written += evaluate_model(model_name, eps, metadata, completed, args.batch_size, device)
            completed = load_completed_records(OUTPUT_PATH)

    print(f"Done. New records written: {total_written}. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
    main()
