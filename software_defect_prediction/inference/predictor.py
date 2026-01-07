import os
import sys
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.serialization
from omegaconf.dictconfig import DictConfig

torch.serialization.add_safe_globals([DictConfig])

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import DefectClassifier
from utils.logging import get_logger

logger = get_logger(__name__)


def _get_device():
    """Определение устройства (CPU/GPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Для Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _load_config(config_path: str):
    """Загрузка конфигурации из файла"""
    import os

    if not os.path.exists(config_path):
        logger.error(f"Конфиг файл не найден: {config_path}")
        return None

    try:
        # Проверяем расширение файла
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            from omegaconf import OmegaConf

            config = OmegaConf.load(config_path)
            logger.info(f"Конфиг загружен из YAML: {config_path}")
            return config
        elif config_path.endswith(".json"):
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Конфиг загружен из JSON: {config_path}")
            return config
        else:
            logger.warning(f"Неизвестный формат конфиг файла: {config_path}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфига {config_path}: {e}")
        return None


class DefectPredictor:
    """Класс для инференса модели"""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.model = self._load_model()
        self.scaler = None
        self.config = None
        self.device = _get_device()

        if config_path:
            _load_config(config_path)

    def _load_model(self):
        """Загрузка модели"""
        logger.info(f"Загрузка модели из {self.model_path}")
        model = DefectClassifier.safe_load_from_checkpoint(self.model_path)
        model.eval()
        return model

    def load_scaler(self, scaler_path: str):
        """Загрузка scaler"""
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, data: pd.DataFrame) -> torch.Tensor:
        """Препроцессинг данных"""
        if self.scaler:
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data.values

        return torch.FloatTensor(data_scaled)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Предсказание дефектов"""
        with torch.no_grad():
            tensor_data = self.preprocess(data)
            logits = self.model(tensor_data)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            return {
                "predictions": preds.cpu().numpy().tolist(),
                "is_defects": (preds == 1).cpu().numpy().tolist(),
                "defect_probabilities": probs[:, 1].cpu().numpy().tolist(),
                "confidences": torch.max(probs, dim=1).values.cpu().numpy().tolist(),
            }

    def predict_batch(self, data_list: pd.DataFrame) -> List[Dict]:
        """Пакетное предсказание"""
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        return results

    def evaluate_on_test(self, test_data_path: str, test_data: pd.DataFrame = None):
        """Оценка модели на тестовых данных"""
        import numpy as np

        if test_data is None and test_data_path:
            # Загружаем данные из файла
            test_data = pd.read_csv(test_data_path)
        elif test_data is None:
            raise ValueError("Необходимо указать test_data_path или передать test_data")

        # Отделяем таргет если есть
        if "defects" in test_data.columns:
            X_test = test_data.drop("defects", axis=1)
            y_test = test_data["defects"].values
        else:
            X_test = test_data
            y_test = None

        # Предсказания
        with torch.no_grad():
            predictions = []
            probabilities = []

            # Для больших данных можно батчами
            batch_size = 32
            for i in range(0, len(X_test), batch_size):
                batch = X_test.iloc[i : i + batch_size]
                tensor_data = self.preprocess(batch)
                logits = self.model(tensor_data)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        # Преобразуем numpy массивы в списки Python
        predictions = [int(p) for p in predictions]  # numpy.int64 -> int
        probabilities = [p.tolist() for p in probabilities]  # numpy.ndarray -> list

        # Если есть истинные метки, вычисляем метрики
        if y_test is not None:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            # Преобразуем y_test в список Python
            y_test_list = [int(y) for y in y_test]

            # Вычисляем метрики и преобразуем в Python float
            metrics = {
                "accuracy": float(accuracy_score(y_test_list, predictions)),
                "precision": float(
                    precision_score(y_test_list, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test_list, predictions, zero_division=0)
                ),  # noqa501
                "f1": float(f1_score(y_test_list, predictions, zero_division=0)),
            }

            # ROC-AUC если есть вероятности
            try:
                probas = np.array(probabilities)[:, 1]  # вероятности класса 1
                metrics["roc_auc"] = float(roc_auc_score(y_test_list, probas))
            except Exception:
                metrics["roc_auc"] = 0.0

            return {
                "metrics": metrics,
                "predictions": predictions,
                "probabilities": probabilities,
                "true_labels": y_test_list,
            }
        else:
            return {"predictions": predictions, "probabilities": probabilities}
