from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import pandas as pd
import json
import os

from .models import AttackInput, AttackResult, TaskInfo, TaskStatus, Sample, Experiment
from ..scheduler.attack_scheduler import AttackScheduler
from src.evaluation.evaluation_engine import (
    get_whitebox_results,
    get_blackbox_results,
    run_blackbox_evaluation,
    run_whitebox_evaluation,
    get_attack_statistics,
    blackbox_evaluator
)

router = APIRouter()
scheduler = AttackScheduler()

experiments_db: Dict[str, Experiment] = {}
samples_db: Dict[str, Sample] = {}

# ========== 样本管理 API ==========

@router.post("/samples", response_model=Sample)
async def create_sample(file: UploadFile = File(...)):
    """上传新样本"""
    sample_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    file_path = f"data/{file.filename}"

    import os
    os.makedirs("data", exist_ok=True)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    sample = Sample(
        id=sample_id,
        name=file.filename,
        type=file.content_type,
        path=file_path,
        created_at=created_at
    )
    samples_db[sample_id] = sample
    return sample

@router.get("/samples", response_model=Dict[str, Sample])
async def list_samples():
    """获取所有样本列表"""
    return samples_db

@router.get("/samples/{sample_id}", response_model=Sample)
async def get_sample(sample_id: str):
    """获取单个样本"""
    sample = samples_db.get(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="样本不存在")
    return sample

@router.delete("/samples/{sample_id}")
async def delete_sample(sample_id: str):
    """删除样本"""
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail="样本不存在")
    del samples_db[sample_id]
    return {"message": "样本已删除"}

# ========== 攻击任务 API ==========

@router.post("/attack/async", response_model=TaskInfo)
async def create_attack_task(
    attack_input: AttackInput,
    background_tasks: BackgroundTasks
):
    """创建异步攻击任务"""
    task_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0.0,
        created_at=created_at,
        updated_at=created_at
    )
    scheduler.add_task(task_id, task_info, attack_input)

    background_tasks.add_task(scheduler.execute_task, task_id)

    return task_info

@router.get("/attack/{task_id}", response_model=TaskInfo)
async def get_attack_task(task_id: str):
    """获取攻击任务状态"""
    task_info = scheduler.get_task(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task_info

@router.post("/attack/sync", response_model=AttackResult)
async def run_attack_sync(attack_input: AttackInput):
    """同步执行攻击任务"""
    try:
        result = await scheduler.execute_attack(attack_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/attack/{task_id}")
async def cancel_attack_task(task_id: str):
    """取消攻击任务"""
    success = scheduler.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"message": "任务已取消"}

@router.get("/attack/list", response_model=Dict[str, TaskInfo])
async def list_attack_tasks():
    """获取所有任务列表"""
    return scheduler.get_all_tasks()

# ========== 黑白盒评估 API ==========

@router.get("/evaluation/whitebox")
async def get_whitebox_evaluation():
    """获取White-box攻击评估结果"""
    results = get_whitebox_results()
    return {"results": results, "count": len(results)}

@router.get("/evaluation/blackbox")
async def get_blackbox_evaluation():
    """获取Black-box迁移攻击评估结果"""
    df = get_blackbox_results()
    if df.empty:
        return {"results": [], "count": 0}

    # 转换为字典列表
    results = df.to_dict('records')
    return {"results": results, "count": len(results)}

@router.get("/evaluation/blackbox/stats")
async def get_blackbox_statistics():
    """获取Black-box攻击统计数据"""
    stats = get_attack_statistics()
    return {"statistics": stats}

@router.post("/evaluation/blackbox/run")
async def run_blackbox_evaluation_api(
    surrogate: str,
    eps: str,
    target_model: Optional[str] = None
):
    """运行Black-box评估"""
    results = run_blackbox_evaluation(surrogate, eps, target_model)
    return {"results": results, "count": len(results)}

@router.get("/evaluation/whitebox/run")
async def run_whitebox_evaluation_api(surrogate: str, eps: str):
    """运行White-box评估"""
    result = run_whitebox_evaluation(surrogate, eps)
    return result

# ========== 实验记录 API ==========

@router.post("/experiments", response_model=Experiment)
async def create_experiment(experiment_name: str, attack_input: AttackInput, attack_result: AttackResult):
    """创建新实验"""
    experiment_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    experiment = Experiment(
        id=experiment_id,
        name=experiment_name,
        config=attack_input.config,
        result=attack_result,
        created_at=created_at
    )
    experiments_db[experiment_id] = experiment
    return experiment

@router.get("/experiments", response_model=Dict[str, Experiment])
async def list_experiments():
    """获取所有实验列表"""
    return experiments_db

@router.get("/experiments/{experiment_id}", response_model=Experiment)
async def get_experiment(experiment_id: str):
    """获取单个实验"""
    experiment = experiments_db.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")
    return experiment

@router.get("/experiments/whitebox")
async def get_whitebox_experiments():
    """获取White-box实验数据"""
    results = get_whitebox_results()
    experiments = []
    for name, data in results.items():
        experiments.append({
            "name": name,
            **data
        })
    return {"experiments": experiments, "count": len(experiments)}

@router.get("/experiments/blackbox")
async def get_blackbox_experiments():
    """获取Black-box实验数据"""
    df = get_blackbox_results()
    if df.empty:
        return {"experiments": [], "count": 0}

    # 按surrogate和eps分组
    grouped = df.groupby(['surrogate', 'eps']).agg({
        'attack_success': ['mean', 'count'],
        'image_id': 'nunique'
    }).reset_index()

    experiments = []
    for _, row in grouped.iterrows():
        exp_name = f"{row['surrogate'][0]}_{row['eps'][0]}"
        experiments.append({
            "name": exp_name,
            "surrogate": row['surrogate'][0],
            "eps": row['eps'][0],
            "total_samples": int(row['image_id']['nunique']),
            "asr": float(row['attack_success']['mean']) * 100
        })

    return {"experiments": experiments, "count": len(experiments)}

# ========== 健康检查 ==========

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}