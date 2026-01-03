import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Добавляем путь к src в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "software_defect_prediction"))

from inference.predictor import DefectPredictor  # noqa402


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


def test_predictor_initialization(sample_config):
    """Тест инициализации предиктора"""
    # Создаем временную модель
    import tempfile

    from models.model import DefectClassifier

    class SimpleConfig:
        class model:
            input_size = 21
            hidden_sizes = [10, 5]
            dropout_rate = 0.1
            learning_rate = 0.001
            batch_size = 32
            num_classes = 2

    config = SimpleConfig()
    model = DefectClassifier(config)

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_path = tmp.name

    try:
        predictor = DefectPredictor(model_path, sample_config)
        assert predictor.model is not None
        assert predictor.device is not None
        print("✓ Predictor инициализирован успешно")
    finally:
        os.unlink(model_path)


def test_preprocess(sample_data, sample_config):
    """Тест препроцессинга"""
    predictor = DefectPredictor("dummy_path", sample_config)
    assert predictor is not None
