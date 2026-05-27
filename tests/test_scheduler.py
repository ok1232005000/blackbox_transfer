import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.api.models import (
    AttackConfig, AttackInput, AttackType, TargetModelType
)
from src.services.scheduler.attack_scheduler import AttackScheduler


@pytest.fixture
def sample_attack_config():
    return AttackConfig(
        attack_type=AttackType.FGSM,
        target_model=TargetModelType.BLACK_BOX,
        epsilon=0.05,
        max_iterations=100,
        learning_rate=0.01,
        confidence=0.9
    )


@pytest.fixture
def sample_attack_input(sample_attack_config):
    return AttackInput(
        data=[
            {"text": "这是一个测试样本"},
            {"text": "这是第二个测试样本"}
        ],
        config=sample_attack_config
    )


@pytest.fixture
def scheduler():
    return AttackScheduler()


def test_attack_config_creation(sample_attack_config):
    assert sample_attack_config.attack_type == AttackType.FGSM
    assert sample_attack_config.epsilon == 0.05
    assert sample_attack_config.max_iterations == 100


def test_attack_input_creation(sample_attack_input):
    assert len(sample_attack_input.data) == 2
    assert sample_attack_input.config.attack_type == AttackType.FGSM


@pytest.mark.asyncio
async def test_scheduler_mock_execution(scheduler, sample_attack_input):
    result = await scheduler.execute_attack(sample_attack_input)

    assert result is not None
    assert hasattr(result, 'success')
    assert hasattr(result, 'adversarial_data')
    assert hasattr(result, 'original_predictions')
    assert hasattr(result, 'adversarial_predictions')
    assert hasattr(result, 'attack_metrics')
    assert hasattr(result, 'execution_time')

    assert len(result.adversarial_data) == len(sample_attack_input.data)
    assert len(result.original_predictions) == len(sample_attack_input.data)
    assert len(result.adversarial_predictions) == len(sample_attack_input.data)


@pytest.mark.asyncio
async def test_scheduler_mock_metrics(scheduler, sample_attack_input):
    result = await scheduler.execute_attack(sample_attack_input)

    assert 'success_rate' in result.attack_metrics
    assert 'average_perturbation' in result.attack_metrics
    assert 'semantic_consistency' in result.attack_metrics
    assert 'transferability' in result.attack_metrics


def test_scheduler_task_management(scheduler):
    from src.services.api.models import TaskInfo, TaskStatus
    from datetime import datetime

    task_id = "test-task-123"
    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0.0,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )

    scheduler.add_task(task_id, task_info, AttackInput(
        data=[{"text": "test"}],
        config=AttackConfig(
            attack_type=AttackType.FGSM,
            target_model=TargetModelType.BLACK_BOX
        )
    ))

    retrieved_task = scheduler.get_task(task_id)
    assert retrieved_task is not None
    assert retrieved_task.task_id == task_id
    assert retrieved_task.status == TaskStatus.PENDING


def test_scheduler_cancel_task(scheduler):
    from src.services.api.models import TaskInfo, TaskStatus
    from datetime import datetime

    task_id = "cancel-test-task"
    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.RUNNING,
        progress=0.5,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )

    scheduler.add_task(task_id, task_info, AttackInput(
        data=[{"text": "test"}],
        config=AttackConfig(
            attack_type=AttackType.FGSM,
            target_model=TargetModelType.BLACK_BOX
        )
    ))

    success = scheduler.cancel_task(task_id)
    assert success is True

    cancelled_task = scheduler.get_task(task_id)
    assert cancelled_task.status == TaskStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
