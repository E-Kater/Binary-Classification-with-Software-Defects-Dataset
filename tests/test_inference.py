import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "software_defect_prediction"))


@pytest.fixture
def sample_data():
    """Тестовые данные"""
    np.random.seed(42)
    n_samples = 10

    data = {
        "loc": np.random.uniform(1, 100, n_samples),
        "v(g)": np.random.uniform(1, 10, n_samples),
        "ev(g)": np.random.uniform(1, 5, n_samples),
        "iv(g)": np.random.uniform(1, 5, n_samples),
        "n": np.random.uniform(10, 100, n_samples),
        "v": np.random.uniform(0.01, 0.5, n_samples),
        "l": np.random.uniform(0.1, 10, n_samples),
        "d": np.random.uniform(1, 50, n_samples),
        "i": np.random.uniform(1, 30, n_samples),
        "e": np.random.uniform(100, 10000, n_samples),
        "b": np.random.uniform(0.01, 0.2, n_samples),
        "t": np.random.uniform(1, 500, n_samples),
        "lOCode": np.random.randint(0, 50, n_samples),
        "lOComment": np.random.randint(0, 50, n_samples),
        "lOBlank": np.random.randint(0, 50, n_samples),
        "locCodeAndComment": np.random.randint(0, 100, n_samples),
        "uniq_Op": np.random.randint(1, 30, n_samples),
        "uniq_Opnd": np.random.randint(1, 30, n_samples),
        "total_Op": np.random.randint(1, 100, n_samples),
        "total_Opnd": np.random.randint(1, 100, n_samples),
        "branchCount": np.random.randint(1, 20, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """Тестовая конфигурация"""
    return {
        "preprocessing": {
            "numeric_features": [
                "loc",
                "v(g)",
                "ev(g)",
                "iv(g)",
                "n",
                "v",
                "l",
                "d",
                "i",
                "e",
                "b",
                "t",
                "lOCode",
                "lOComment",
                "lOBlank",
                "locCodeAndComment",
                "uniq_Op",
                "uniq_Opnd",
                "total_Op",
                "total_Opnd",
                "branchCount",
            ]
        }
    }


# Вставьте в ваш test_inference.py после существующих тестов


def test_model_forward_pass():
    """Тест forward pass модели"""
    from types import SimpleNamespace

    from models.model import DefectClassifier

    # Создаем конфигурацию
    config = SimpleNamespace()
    config.model = SimpleNamespace()
    config.model.input_size = 21
    config.model.hidden_sizes = [64, 32]
    config.model.dropout_rate = 0.2
    config.model.learning_rate = 0.001
    config.model.batch_size = 32
    config.model.num_classes = 2

    # Создаем модель
    model = DefectClassifier(config)

    # Тестовый вход
    test_input = torch.randn(4, 21)

    # Проверяем forward pass
    output = model(test_input)
    assert output.shape == (4, 2)
    assert not torch.isnan(output).any()


def test_model_training_step():
    """Тест training step"""
    from types import SimpleNamespace

    from models.model import DefectClassifier

    config = SimpleNamespace()
    config.model = SimpleNamespace()
    config.model.input_size = 5  # Уменьшаем для теста
    config.model.hidden_sizes = [10, 5]
    config.model.dropout_rate = 0.1
    config.model.learning_rate = 0.001
    config.model.batch_size = 32
    config.model.num_classes = 2

    model = DefectClassifier(config)

    # Создаем тестовый батч
    x = torch.randn(8, 5)
    y = torch.randint(0, 2, (8,))

    # Выполняем training step
    loss = model.training_step((x, y), batch_idx=0)

    assert loss is not None
    assert loss.item() > 0


def test_model_validation_step():
    """Тест validation step"""
    from types import SimpleNamespace

    from models.model import DefectClassifier

    config = SimpleNamespace()
    config.model = SimpleNamespace()
    config.model.input_size = 21
    config.model.hidden_sizes = [10, 5]
    config.model.dropout_rate = 0.1
    config.model.learning_rate = 0.001
    config.model.batch_size = 32
    config.model.num_classes = 2

    model = DefectClassifier(config)

    # Создаем тестовый батч
    x = torch.randn(8, 21)
    y = torch.randint(0, 2, (8,))

    # Выполняем validation step
    model.training_step((x, y), batch_idx=0)
    loss = model.validation_step((x, y), batch_idx=0)

    assert loss is not None
    assert isinstance(loss, torch.Tensor)


def test_model_configure_optimizers():
    """Тест конфигурации оптимизаторов"""
    from types import SimpleNamespace

    from models.model import DefectClassifier

    config = SimpleNamespace()
    config.model = SimpleNamespace()
    config.model.input_size = 5
    config.model.hidden_sizes = [10, 5]
    config.model.dropout_rate = 0.1
    config.model.learning_rate = 0.01  # Явное значение для теста
    config.model.batch_size = 32
    config.model.num_classes = 2

    model = DefectClassifier(config)

    optimizers_config = model.configure_optimizers()

    assert "optimizer" in optimizers_config
    assert isinstance(optimizers_config["optimizer"], torch.optim.Adam)

    # Проверяем learning rate
    optimizer = optimizers_config["optimizer"]
    assert optimizer.param_groups[0]["lr"] == 0.01
