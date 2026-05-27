import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from src.services.api.models import AttackInput, AttackResult, TaskInfo, TaskStatus
from src.common.logger import get_logger
from src.common.config import config
import importlib

logger = get_logger(__name__)

class AttackScheduler:
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.attack_inputs: Dict[str, AttackInput] = {}
        self._load_modules()
    
    def _load_modules(self):
        """动态加载攻击生成模块和评估模块"""
        try:
            self.attack_module = importlib.import_module(config["paths"]["attack_module"])
            logger.info("攻击生成模块加载成功")
        except ImportError:
            logger.warning("攻击生成模块未找到，使用模拟模块")
            self.attack_module = None
        
        try:
            self.evaluation_module = importlib.import_module(config["paths"]["evaluation_module"])
            logger.info("评估模块加载成功")
        except ImportError:
            logger.warning("评估模块未找到，使用模拟模块")
            self.evaluation_module = None
    
    def add_task(self, task_id: str, task_info: TaskInfo, attack_input: AttackInput):
        """添加任务到调度器"""
        self.tasks[task_id] = task_info
        self.attack_inputs[task_id] = attack_input
        logger.info(f"任务 {task_id} 已添加")
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """获取所有任务"""
        return self.tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.FAILED
                task.progress = 0.0
                task.updated_at = datetime.now().isoformat()
                return True
        return False
    
    async def execute_task(self, task_id: str):
        """执行任务"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        attack_input = self.attack_inputs.get(task_id)
        
        try:
            task.status = TaskStatus.RUNNING
            task.progress = 0.2
            task.updated_at = datetime.now().isoformat()
            logger.info(f"任务 {task_id} 开始执行")
            
            result = await self.execute_attack(attack_input)
            
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.result = result
            task.updated_at = datetime.now().isoformat()
            logger.info(f"任务 {task_id} 执行完成")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.progress = 0.0
            task.updated_at = datetime.now().isoformat()
            logger.error(f"任务 {task_id} 执行失败: {str(e)}")
    
    async def execute_attack(self, attack_input: AttackInput) -> AttackResult:
        """执行攻击流程"""
        import time
        start_time = time.time()
        
        # 1. 调用攻击生成模块生成对抗样本
        logger.info("开始生成对抗样本")
        adversarial_data, original_predictions, adversarial_predictions = \
            await self._generate_adversarial_samples(attack_input)
        
        # 2. 调用评估模块计算指标
        logger.info("开始评估攻击效果")
        attack_metrics = await self._evaluate_attack(
            attack_input.data, 
            adversarial_data, 
            original_predictions, 
            adversarial_predictions
        )
        
        # 3. 判断攻击是否成功
        success_count = sum(
            1 for orig, adv in zip(original_predictions, adversarial_predictions)
            if orig != adv
        )
        success = success_count > 0
        
        execution_time = time.time() - start_time
        
        return AttackResult(
            success=success,
            adversarial_data=adversarial_data,
            original_predictions=original_predictions,
            adversarial_predictions=adversarial_predictions,
            attack_metrics=attack_metrics,
            execution_time=execution_time
        )
    
    async def _generate_adversarial_samples(self, attack_input: AttackInput) -> tuple:
        """调用攻击生成模块"""
        if self.attack_module and hasattr(self.attack_module, 'generate_adversarial'):
            return await self.attack_module.generate_adversarial(attack_input.config, attack_input.data)
        else:
            return self._mock_generate_adversarial(attack_input)
    
    def _mock_generate_adversarial(self, attack_input: AttackInput) -> tuple:
        """模拟攻击生成（用于测试）"""
        import random
        
        adversarial_data = []
        original_predictions = []
        adversarial_predictions = []
        
        for item in attack_input.data:
            perturbed_item = {**item, "perturbed": True}
            if "text" in item:
                perturbed_item["text"] = item["text"] + " [对抗扰动]"
            if "image" in item:
                perturbed_item["image"] = "perturbed_" + item["image"]
            
            adversarial_data.append(perturbed_item)
            original_predictions.append(random.randint(0, 9))
            # 模拟攻击成功（部分样本标签改变）
            if random.random() > 0.3:
                adversarial_predictions.append(random.randint(0, 9))
            else:
                adversarial_predictions.append(original_predictions[-1])
        
        return adversarial_data, original_predictions, adversarial_predictions
    
    async def _evaluate_attack(self, original_data: list, adversarial_data: list,
                                original_preds: list, adversarial_preds: list) -> dict:
        """调用评估模块"""
        if self.evaluation_module and hasattr(self.evaluation_module, 'evaluate'):
            return await self.evaluation_module.evaluate(
                original_data, adversarial_data, original_preds, adversarial_preds
            )
        else:
            return self._mock_evaluate(original_preds, adversarial_preds)
    
    def _mock_evaluate(self, original_preds: list, adversarial_preds: list) -> dict:
        """模拟评估指标"""
        total = len(original_preds)
        success_count = sum(1 for o, a in zip(original_preds, adversarial_preds) if o != a)
        
        return {
            "success_rate": success_count / total if total > 0 else 0.0,
            "average_perturbation": 0.03,
            "semantic_consistency": 0.85,
            "transferability": 0.72
        }