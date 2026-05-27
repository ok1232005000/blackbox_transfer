from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class AttackType(str, Enum):
    FGSM = "fgsm"
    PGD = "pgd"
    TEXT_REPLACEMENT = "text_replacement"
    CW = "cw"

class TargetModelType(str, Enum):
    BLACK_BOX = "black_box"
    WHITE_BOX = "white_box"

class AttackConfig(BaseModel):
    attack_type: AttackType = Field(..., description="攻击类型")
    target_model: TargetModelType = Field(..., description="目标模型类型")
    epsilon: float = Field(0.05, description="扰动强度", ge=0.0, le=1.0)
    max_iterations: int = Field(100, description="最大迭代次数", ge=1)
    learning_rate: float = Field(0.01, description="学习率", ge=0.001, le=0.1)
    target_label: Optional[int] = Field(None, description="目标标签（可选）")
    confidence: float = Field(0.9, description="攻击置信度阈值", ge=0.0, le=1.0)

class AttackInput(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="输入数据列表")
    config: AttackConfig = Field(..., description="攻击配置")

class AttackResult(BaseModel):
    success: bool = Field(..., description="攻击是否成功")
    adversarial_data: List[Dict[str, Any]] = Field(..., description="生成的对抗样本")
    original_predictions: List[int] = Field(..., description="原始预测结果")
    adversarial_predictions: List[int] = Field(..., description="对抗样本预测结果")
    attack_metrics: Dict[str, float] = Field(..., description="攻击指标")
    execution_time: float = Field(..., description="执行时间（秒）")

class EvaluationResult(BaseModel):
    attack_success_rate: float = Field(..., description="攻击成功率")
    average_perturbation: float = Field(..., description="平均扰动强度")
    semantic_consistency: float = Field(..., description="语义一致性")
    transferability: float = Field(..., description="迁移能力")
    detailed_results: List[Dict[str, Any]] = Field(..., description="详细结果")

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Sample(BaseModel):
    id: str = Field(..., description="样本ID")
    name: str = Field(..., description="样本名称")
    type: str = Field(..., description="样本类型")
    path: str = Field(..., description="存储路径")
    created_at: str = Field(..., description="创建时间")

class Experiment(BaseModel):
    id: str = Field(..., description="实验ID")
    name: str = Field(..., description="实验名称")
    config: AttackConfig = Field(..., description="攻击配置")
    result: AttackResult = Field(..., description="攻击结果")
    created_at: str = Field(..., description="创建时间")

class TaskInfo(BaseModel):
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: float = Field(0.0, description="任务进度", ge=0.0, le=1.0)
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    result: Optional[AttackResult] = Field(None, description="任务结果")