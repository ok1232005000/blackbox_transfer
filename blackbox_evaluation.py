import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from datetime import datetime

# --- Configuration ---
BASE_DIR = 'd:\\java\\javacode\\weilai'
ADVERSARIAL_DIR = os.path.join(BASE_DIR, 'adversarial')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.json')
RECORDS_PATH = os.path.join(BASE_DIR, 'experiments', 'records.jsonl')

SURROGATE_CONFIGS = {
    'resnet_only': ['eps4', 'eps8', 'eps16'],
    'vit_only': ['eps4', 'eps8', 'eps16']
}

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model and Transforms ---
# Standard transform for most models
transform_standard = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transform for InceptionV3
transform_inception = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading pre-trained models...")
TARGET_MODELS = {
    'inception_v3': {'model': models.inception_v3(pretrained=True), 'transform': transform_inception},
    'densenet121': {'model': models.densenet121(pretrained=True), 'transform': transform_standard},
    'mobilenet_v2': {'model': models.mobilenet_v2(pretrained=True), 'transform': transform_standard},
    'efficientnet_b0': {'model': models.efficientnet_b0(pretrained=True), 'transform': transform_standard},
}

# Set models to eval mode and move to GPU
for model_info in TARGET_MODELS.values():
    model_info['model'].to(device)
    model_info['model'].eval()
print("Models loaded and moved to device.")

# --- Load Metadata ---
print("Loading metadata...")
with open(METADATA_PATH, 'r') as f:
    metadata_list = json.load(f)
metadata_map = {item['id']: item for item in metadata_list}
print("Metadata loaded.")

# --- Main Evaluation Loop ---
# Ensure the output directory exists
os.makedirs(os.path.dirname(RECORDS_PATH), exist_ok=True)

print("Starting black-box evaluation...")
with open(RECORDS_PATH, 'a') as records_file:
    for surrogate, eps_values in SURROGATE_CONFIGS.items():
        for eps in eps_values:
            image_dir = os.path.join(ADVERSARIAL_DIR, surrogate, eps)
            if not os.path.isdir(image_dir):
                print(f"Warning: Directory not found, skipping: {image_dir}")
                continue

            print(f"Processing directory: {image_dir}")
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            total_images = len(image_files)
            for i, image_filename in enumerate(image_files):
                
                image_id = os.path.splitext(image_filename)[0]
                image_path = os.path.join(image_dir, image_filename)
                
                if image_id not in metadata_map:
                    continue
                
                true_label = metadata_map[image_id]['true_label']
                
                try:
                    adv_image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Error opening image {image_path}: {e}")
                    continue

                for model_name, model_info in TARGET_MODELS.items():
                    model = model_info['model']
                    transform = model_info['transform']
                    
                    input_tensor = transform(adv_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        # Handle InceptionV3's output which might be an object
                        if model_name == 'inception_v3':
                            if isinstance(output, models.inception.InceptionOutputs):
                                output = output.logits
                        _, adv_pred = torch.max(output, 1)
                        adv_output = adv_pred.item()

                    attack_success = adv_output != true_label

                    record = {
                        "image_id": image_id,
                        "eps": eps,
                        "surrogate": surrogate,
                        "target_model": model_name,
                        "clean_output": None, # Clean images are not available
                        "adv_output": adv_output,
                        "semantic_sim": None, # Cannot be calculated without clean image
                        "attack_success": attack_success,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                    
                    records_file.write(json.dumps(record) + '\n')
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{total_images} images in {eps} folder.")

print("Black-box evaluation complete. Results saved to experiments/records.jsonl")
