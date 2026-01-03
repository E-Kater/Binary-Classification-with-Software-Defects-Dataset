import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

# Добавляем путь в sys.path для корректных импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import DefectClassifier  # noqa402


@pytest.fixture
def sample_config():
    config = OmegaConf.create(
        {
            "model": {
                "input_size": 21,
                "hidden_sizes": [10, 5],
                "dropout_rate": 0.1,
                "learning_rate": 0.001,
                "batch_size": 16,
                "num_classes": 2,
            },
            "data": {
                "raw_path": "data/raw/defects.csv",
                "test_size": 0.2,
                "val_size": 0.1,
                "random_state": 42,
                "target_col": "defects",
            },
            "preprocessing": {
                "numeric_features": ["loc", "v(g)"],
                "categorical_features": [],
                "scaler": "standard",
            },
        }
    )
    return config


def test_model_creation(sample_config):
    """Тест создания модели"""
    model = DefectClassifier(sample_config)
    assert isinstance(model, DefectClassifier)

    # Проверка архитектуры
    test_input = torch.randn(4, 21)
    output = model(test_input)
    assert output.shape == (4, 2)


def test_model_forward(sample_config):
    """Тест forward pass"""
    model = DefectClassifier(sample_config)
    x = torch.randn(8, 21)
    output = model(x)
    assert output.shape == (8, 2)
