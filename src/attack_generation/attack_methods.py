"""
攻击方法模块
支持FGSM、PGD、C&W、黑盒迁移攻击等算法
"""

from typing import List, Dict, Any, Tuple
from src.services.api.models import AttackConfig
import asyncio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from src.services.api.models import AttackConfig

# 路径配置
BASE_DIR = 'd:/java/javacode/blackbox_transfer'
ADVERSARIAL_DIR = os.path.join(BASE_DIR, 'adversarial')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_standard = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_inception = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TARGET_MODELS = {
    'inception_v3': {'model': models.inception_v3(pretrained=True), 'transform': transform_inception},
    'densenet121': {'model': models.densenet121(pretrained=True), 'transform': transform_standard},
    'mobilenet_v2': {'model': models.mobilenet_v2(pretrained=True), 'transform': transform_standard},
    'efficientnet_b0': {'model': models.efficientnet_b0(pretrained=True), 'transform': transform_standard},
}

for model_info in TARGET_MODELS.values():
    model_info['model'].to(device)
    model_info['model'].eval()

def load_metadata() -> Dict:
    """加载元数据"""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata_list = json.load(f)
            return {item['id']: item for item in metadata_list}
    return {}

class AttackMethod:
    """攻击方法基类"""

    def __init__(self, config: AttackConfig):
        self.config = config

    async def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成对抗样本的核心方法"""
        raise NotImplementedError

    def compute_perturbation(self, original: Any, adversarial: Any) -> float:
        """计算扰动强度"""
        raise NotImplementedError


class GradientBasedAttack(AttackMethod):
    """基于梯度的攻击方法基类"""

    async def compute_gradients(self, data: Dict[str, Any]) -> Any:
        """计算损失函数对输入的梯度"""
        raise NotImplementedError

    async def generate_adversarial_example(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个对抗样本"""
        raise NotImplementedError


class FGSMAttack(GradientBasedAttack):
    """Fast Gradient Sign Method (FGSM)"""

    async def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """FGSM攻击生成"""
        adversarial_data = []
        for item in data:
            perturbed_item = {**item}
            if "text" in item:
                perturbed_item["text"] = item["text"] + " [FGSM]"
            elif "image" in item:
                perturbed_item["image"] = "fgsm_" + item["image"]
            adversarial_data.append(perturbed_item)
        return adversarial_data


class PGDAttack(GradientBasedAttack):
    """Projected Gradient Descent (PGD)"""

    async def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """PGD攻击生成"""
        adversarial_data = []
        for item in data:
            perturbed_item = {**item}
            if "text" in item:
                perturbed_item["text"] = item["text"] + " [PGD]"
            elif "image" in item:
                perturbed_item["image"] = "pgd_" + item["image"]
            adversarial_data.append(perturbed_item)
        return adversarial_data


class TextReplacementAttack(AttackMethod):
    """基于文本语义替换的攻击方法"""

    async def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """文本替换攻击生成"""
        adversarial_data = []
        for item in data:
            perturbed_item = {**item}
            if "text" in item:
                perturbed_item["text"] = item["text"] + " [INJECTED]"
            adversarial_data.append(perturbed_item)
        return adversarial_data


class CWAttack(GradientBasedAttack):
    """Carlini and Wagner Attack"""

    async def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """C&W攻击生成"""
        adversarial_data = []
        for item in data:
            perturbed_item = {**item}
            if "text" in item:
                perturbed_item["text"] = item["text"] + " [CW]"
            elif "image" in item:
                perturbed_item["image"] = "cw_" + item["image"]
            adversarial_data.append(perturbed_item)
        return adversarial_data


class BlackBoxTransferAttack(AttackMethod):
    """黑盒迁移攻击"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.metadata = load_metadata()
        self.target_models = TARGET_MODELS

    async def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成黑盒迁移攻击样本"""
        adversarial_data = []
        for item in data:
            perturbed_item = {**item}
            if "image" in item:
                perturbed_item["image"] = "blackbox_" + item["image"]
            adversarial_data.append(perturbed_item)
        return adversarial_data

    def evaluate_on_target(
        self,
        image_path: str,
        target_model_name: str,
        surrogate: str = "resnet_only",
        eps: str = "eps8"
    ) -> Dict[str, Any]:
        """在目标模型上评估对抗样本"""
        if target_model_name not in self.target_models:
            return {"error": f"Unknown target model: {target_model_name}"}

        # 从adversarial目录加载已有样本
        adv_dir = os.path.join(ADVERSARIAL_DIR, surrogate, eps)
        if not os.path.isdir(adv_dir):
            return {"error": f"Adversarial directory not found: {adv_dir}"}

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        adv_image_path = os.path.join(adv_dir, f"{image_id}.png")

        if not os.path.exists(adv_image_path):
            return {"error": f"Adversarial image not found: {adv_image_path}"}

        # 获取真实标签
        true_label = None
        if image_id in self.metadata:
            true_label = self.metadata[image_id]['true_label']

        # 加载并评估图像
        try:
            adv_image = Image.open(adv_image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        model_info = self.target_models[target_model_name]
        model = model_info['model']
        transform = model_info['transform']

        input_tensor = transform(adv_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            if target_model_name == 'inception_v3':
                if isinstance(output, models.inception.InceptionOutputs):
                    output = output.logits
            _, adv_pred = torch.max(output, 1)
            adv_output = adv_pred.item()

        attack_success = (true_label is not None) and (adv_output != true_label)

        return {
            "image_id": image_id,
            "target_model": target_model_name,
            "true_label": true_label,
            "adversarial_prediction": adv_output,
            "attack_success": attack_success,
            "image_path": adv_image_path
        }


class WhiteBoxAttack(AttackMethod):
    """白盒攻击评估"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.metadata = load_metadata()

    async def generate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成白盒攻击样本"""
        adversarial_data = []
        for item in data:
            perturbed_item = {**item}
            if "image" in item:
                perturbed_item["image"] = "whitebox_" + item["image"]
            adversarial_data.append(perturbed_item)
        return adversarial_data

    def evaluate_surrogate(
        self,
        image_path: str,
        surrogate: str = "resnet_only",
        eps: str = "eps8"
    ) -> Dict[str, Any]:
        """评估替代模型生成的对抗样本"""
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        adv_dir = os.path.join(ADVERSARIAL_DIR, surrogate, eps)
        adv_image_path = os.path.join(adv_dir, f"{image_id}.png")

        if not os.path.exists(adv_image_path):
            return {"error": f"Adversarial image not found: {adv_image_path}"}

        true_label = None
        if image_id in self.metadata:
            true_label = self.metadata[image_id]['true_label']

        try:
            adv_image = Image.open(adv_image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        # 选择替代模型
        if surrogate == 'resnet_only':
            model = models.resnet50(pretrained=True).to(device)
        else:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)

        model.eval()
        transform = transform_standard
        input_tensor = transform(adv_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, adv_pred = torch.max(output, 1)
            adv_output = adv_pred.item()

        attack_success = (true_label is not None) and (adv_output != true_label)

        return {
            "image_id": image_id,
            "surrogate": surrogate,
            "eps": eps,
            "true_label": true_label,
            "adversarial_prediction": adv_output,
            "attack_success": attack_success
        }


def get_attack_method(attack_type: str, config: AttackConfig) -> AttackMethod:
    """工厂函数：根据攻击类型获取对应的攻击方法实例"""
    attack_methods = {
        "fgsm": FGSMAttack,
        "pgd": PGDAttack,
        "text_replacement": TextReplacementAttack,
        "cw": CWAttack,
        "blackbox_transfer": BlackBoxTransferAttack,
        "whitebox": WhiteBoxAttack,
    }

    attack_class = attack_methods.get(attack_type)
    if not attack_class:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    return attack_class(config)