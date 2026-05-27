"""
黑白盒攻击评估模块
整合blackbox_evaluation.py和analyze_results.py的核心功能
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from src.services.api.models import AttackConfig

# 配置路径
BASE_DIR = 'd:/java/javacode/blackbox_transfer'
ADVERSARIAL_DIR = os.path.join(BASE_DIR, 'adversarial')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata(2).json')
RECORDS_PATH = os.path.join(BASE_DIR, 'experiments2', 'records.jsonl')
WHITEBOX_PATH = os.path.join(BASE_DIR, 'experiments2', 'whitebox_results.json')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis_results', 'experiments2')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型和变换配置
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

# 目标模型映射
TARGET_MODELS = {
    'inception_v3': {'model': models.inception_v3(pretrained=True), 'transform': transform_inception},
    'densenet121': {'model': models.densenet121(pretrained=True), 'transform': transform_standard},
    'mobilenet_v2': {'model': models.mobilenet_v2(pretrained=True), 'transform': transform_standard},
    'efficientnet_b0': {'model': models.efficientnet_b0(pretrained=True), 'transform': transform_standard},
}

# 初始化模型
for model_info in TARGET_MODELS.values():
    model_info['model'].to(device)
    model_info['model'].eval()

# 元数据缓存
_metadata_cache = None

def get_metadata() -> Dict:
    """获取元数据缓存"""
    global _metadata_cache
    if _metadata_cache is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata_list = json.load(f)
                _metadata_cache = {item['id']: item for item in metadata_list}
        else:
            _metadata_cache = {}
    return _metadata_cache

class WhiteBoxEvaluator:
    """White-box攻击评估器"""

    def __init__(self):
        self.metadata = get_metadata()
        self.surrogate_configs = {
            'resnet_only': ['eps4', 'eps8', 'eps16'],
            'vit_only': ['eps4', 'eps8', 'eps16']
        }

    def evaluate(self, surrogate: str, eps: str) -> Dict:
        """评估指定配置的white-box攻击"""
        results = {
            'surrogate': surrogate,
            'eps': eps,
            'total': 0,
            'whitebox_asr': 0.0
        }

        image_dir = os.path.join(ADVERSARIAL_DIR, surrogate, eps)
        if not os.path.isdir(image_dir):
            return results

        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        results['total'] = len(image_files)

        if results['total'] == 0:
            return results

        success_count = 0
        for image_filename in image_files:
            image_id = os.path.splitext(image_filename)[0]
            if image_id not in self.metadata:
                continue

            true_label = self.metadata[image_id]['true_label']

            try:
                adv_image = Image.open(os.path.join(image_dir, image_filename)).convert('RGB')
            except:
                continue

            # 使用surrogate模型预测
            if surrogate == 'resnet_only':
                model = models.resnet50(pretrained=True).to(device)
                transform = transform_standard
            else:
                model = models.vit_b_16(pretrained=True).to(device)
                transform = transform_standard

            model.eval()
            input_tensor = transform(adv_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, adv_pred = torch.max(output, 1)
                adv_output = adv_pred.item()

            if adv_output != true_label:
                success_count += 1

        results['whitebox_asr'] = success_count / results['total'] if results['total'] > 0 else 0.0
        return results

    def evaluate_all(self) -> Dict:
        """评估所有配置"""
        all_results = {}
        for surrogate, eps_values in self.surrogate_configs.items():
            for eps in eps_values:
                key = f"{surrogate}_{eps}"
                all_results[key] = self.evaluate(surrogate, eps)

        # 保存结果
        os.makedirs(os.path.dirname(WHITEBOX_PATH), exist_ok=True)
        with open(WHITEBOX_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)

        return all_results

class BlackBoxEvaluator:
    """Black-box迁移攻击评估器"""

    def __init__(self):
        self.metadata = get_metadata()
        self.target_models = TARGET_MODELS

    def evaluate(
        self,
        surrogate: str,
        eps: str,
        target_model_name: str = None
    ) -> List[Dict]:
        """评估指定配置的black-box迁移攻击"""
        results = []
        image_dir = os.path.join(ADVERSARIAL_DIR, surrogate, eps)

        if not os.path.isdir(image_dir):
            return results

        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        target_models_to_eval = (
            {target_model_name: self.target_models[target_model_name]}
            if target_model_name and target_model_name in self.target_models
            else self.target_models
        )

        for image_filename in image_files:
            image_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(image_dir, image_filename)

            if image_id not in self.metadata:
                continue

            true_label = self.metadata[image_id]['true_label']

            try:
                adv_image = Image.open(image_path).convert('RGB')
            except:
                continue

            for model_name, model_info in target_models_to_eval.items():
                model = model_info['model']
                transform = model_info['transform']

                input_tensor = transform(adv_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
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
                    "adv_output": adv_output,
                    "attack_success": attack_success,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                results.append(record)

        return results

    def evaluate_all(self) -> pd.DataFrame:
        """评估所有配置并保存"""
        all_records = []
        surrogate_configs = {
            'resnet_only': ['eps4', 'eps8', 'eps16'],
            'vit_only': ['eps4', 'eps8', 'eps16']
        }

        for surrogate, eps_values in surrogate_configs.items():
            for eps in eps_values:
                records = self.evaluate(surrogate, eps)
                all_records.extend(records)

        # 保存结果
        os.makedirs(os.path.dirname(RECORDS_PATH), exist_ok=True)
        with open(RECORDS_PATH, 'w') as f:
            for record in all_records:
                f.write(json.dumps(record) + '\n')

        return pd.DataFrame(all_records)

class AttackAnalyzer:
    """攻击结果分析器"""

    def __init__(self):
        self.analysis_dir = ANALYSIS_DIR
        os.makedirs(self.analysis_dir, exist_ok=True)

    def load_whitebox_results(self) -> Dict:
        """加载white-box结果"""
        if os.path.exists(WHITEBOX_PATH):
            with open(WHITEBOX_PATH, 'r') as f:
                return json.load(f)
        return {}

    def load_blackbox_results(self) -> pd.DataFrame:
        """加载black-box结果"""
        if os.path.exists(RECORDS_PATH):
            return pd.read_json(RECORDS_PATH, lines=True)
        return pd.DataFrame()

    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """计算统计数据"""
        if df.empty:
            return {}

        stats = {
            'total_samples': len(df),
            'overall_asr': df['attack_success'].mean() * 100,
            'by_surrogate': df.groupby('surrogate')['attack_success'].mean() * 100,
            'by_eps': df.groupby('eps')['attack_success'].mean() * 100,
            'by_target': df.groupby('target_model')['attack_success'].mean() * 100,
        }

        return stats

# 全局评估器实例
whitebox_evaluator = WhiteBoxEvaluator()
blackbox_evaluator = BlackBoxEvaluator()
analyzer = AttackAnalyzer()

def get_whitebox_results() -> Dict:
    """获取white-box攻击结果"""
    return analyzer.load_whitebox_results()

def get_blackbox_results() -> pd.DataFrame:
    """获取black-box攻击结果"""
    return analyzer.load_blackbox_results()

def run_blackbox_evaluation(surrogate: str, eps: str, target_model: str = None) -> List[Dict]:
    """运行black-box评估"""
    return blackbox_evaluator.evaluate(surrogate, eps, target_model)

def run_whitebox_evaluation(surrogate: str, eps: str) -> Dict:
    """运行white-box评估"""
    return whitebox_evaluator.evaluate(surrogate, eps)

def get_attack_statistics() -> Dict:
    """获取攻击统计信息"""
    df = analyzer.load_blackbox_results()
    return analyzer.compute_statistics(df)
