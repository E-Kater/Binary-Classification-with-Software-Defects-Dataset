import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "software_defect_prediction"))

from inference.predictor import DefectPredictor


class MockModel:
    def __init__(self):
        self.eval_called = False
        self.device = torch.device("cpu")

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, x):
        # Возвращаем реалистичные логгиты
        batch_size = x.shape[0]
        # Создаем тензор с реалистичными значениями
        logits = torch.randn(batch_size, 2)  # 2 класса
        return logits

    def to(self, device):
        self.device = device
        return self


class TestDefectPredictor:
    """Класс с тестами для DefectPredictor"""

    @pytest.fixture
    def sample_dataframe(self):
        """Фикстура с тестовыми данными"""
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
    def mock_model(self):
        """Фикстура с мок-моделью"""
        model = Mock()

        # Мокаем forward pass
        def mock_forward(x):
            # Возвращаем случайные логгиты
            batch_size = x.shape[0]
            logits = torch.randn(batch_size, 2)  # 2 класса
            return logits

        model.forward = mock_forward
        model.eval = Mock()
        model.cpu = Mock(return_value=model)
        model.to = Mock(return_value=model)

        return model

    def test_initialization_without_config(self, mock_model):
        """Тест 1: Инициализация без конфигурационного файла"""
        # Мокаем загрузку модели
        with patch(
            "inference.predictor.DefectClassifier.safe_load_from_checkpoint"
        ) as mock_load:
            mock_load.return_value = mock_model

            # Создаем временный файл модели
            with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
                model_path = tmp.name

            try:
                # Инициализируем предиктор без конфига
                predictor = DefectPredictor(model_path)

                # Проверяем атрибуты
                assert predictor.model_path == model_path
                assert predictor.model is not None
                assert predictor.device is not None
                assert predictor.scaler is None
                assert predictor.config is None

                # Проверяем, что модель переведена в eval режим
                mock_model.eval.assert_called_once()

                print("✓ Тест 1 пройден: Инициализация без конфига работает")

            finally:
                import os

                if os.path.exists(model_path):
                    os.unlink(model_path)

    def test_initialization_with_config(self, mock_model):
        """Тест 2: Инициализация с конфигурационным файлом"""
        # Мокаем загрузку модели и конфига
        with patch(
            "inference.predictor.DefectClassifier.safe_load_from_checkpoint"
        ) as mock_load, patch("inference.predictor._load_config") as mock_load_config:
            mock_load.return_value = mock_model
            mock_load_config.return_value = {"some": "config"}

            # Создаем временные файлы
            with tempfile.NamedTemporaryFile(
                suffix=".ckpt", delete=False
            ) as model_tmp, tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False
            ) as config_tmp:
                model_path = model_tmp.name
                config_path = config_tmp.name

            try:
                # Инициализируем предиктор с конфигом
                predictor = DefectPredictor(model_path, config_path)

                # Проверяем атрибуты
                assert predictor.model_path == model_path
                assert predictor.model is not None

                # Проверяем, что конфиг был загружен
                mock_load_config.assert_called_once_with(config_path)

                print("✓ Тест 2 пройден: Инициализация с конфигом работает")

            finally:
                import os

                if os.path.exists(model_path):
                    os.unlink(model_path)
                if os.path.exists(config_path):
                    os.unlink(config_path)

    def test_load_scaler(self):
        """Тест 3: Загрузка scaler"""
        # Создаем временный scaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(10, 21))

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            scaler_path = tmp.name
            joblib.dump(scaler, scaler_path)

        try:
            # Мокаем загрузку модели
            with patch("inference.predictor.DefectClassifier.safe_load_from_checkpoint"):
                predictor = DefectPredictor("dummy.ckpt")

                # Загружаем scaler
                predictor.load_scaler(scaler_path)

                # Проверяем, что scaler загружен
                assert predictor.scaler is not None
                assert hasattr(predictor.scaler, "transform")

                print("✓ Тест 3 пройден: Scaler загружается корректно")

        finally:
            import os

            if os.path.exists(scaler_path):
                os.unlink(scaler_path)

    def test_preprocess_without_scaler(self, sample_dataframe, mock_model):
        """Тест 4: Препроцессинг без scaler"""
        # Мокаем загрузку модели
        with patch(
            "inference.predictor.DefectClassifier.safe_load_from_checkpoint"
        ) as mock_load:
            mock_load.return_value = mock_model

            predictor = DefectPredictor("dummy.ckpt")
            predictor.scaler = None  # Убеждаемся, что scaler не установлен

            # Выполняем препроцессинг
            tensor_data = predictor.preprocess(sample_dataframe)

            # Проверяем результат
            assert isinstance(tensor_data, torch.Tensor)
            assert tensor_data.dtype == torch.float32
            assert tensor_data.shape == (len(sample_dataframe), 21)

            # Без scaler данные не должны масштабироваться
            # (проверяем, что значения примерно те же)
            expected_data = torch.FloatTensor(sample_dataframe.values)
            assert torch.allclose(tensor_data, expected_data, rtol=1e-5)

            print("✓ Тест 4 пройден: Препроцессинг без scaler работает")

    def test_preprocess_with_scaler(self, sample_dataframe, mock_model):
        """Тест 5: Препроцессинг с scaler"""
        # Создаем тестовый scaler
        scaler = StandardScaler()
        scaler.fit(sample_dataframe.values)

        # Мокаем загрузку модели
        with patch(
            "inference.predictor.DefectClassifier.safe_load_from_checkpoint"
        ) as mock_load:
            mock_load.return_value = mock_model

            predictor = DefectPredictor("dummy.ckpt")
            predictor.scaler = scaler

            # Выполняем препроцессинг
            tensor_data = predictor.preprocess(sample_dataframe)

            # Проверяем результат
            assert isinstance(tensor_data, torch.Tensor)
            assert tensor_data.dtype == torch.float32
            assert tensor_data.shape == (len(sample_dataframe), 21)

            # Проверяем, что данные масштабированы
            # (после StandardScaler среднее должно быть около 0)
            assert abs(tensor_data.mean().item()) < 1.0

            print("✓ Тест 5 пройден: Препроцессинг с scaler работает")

    def test_predict_basic(self, sample_dataframe):
        """Тест 6: Базовое предсказание"""

        mock_model = MockModel()

        # Мокаем загрузку модели
        with patch(
            "inference.predictor.DefectClassifier.safe_load_from_checkpoint"
        ) as mock_load:
            mock_load.return_value = mock_model

            predictor = DefectPredictor("dummy.ckpt")
            predictor.scaler = None  # Без scaler для простоты

            # Выполняем предсказание
            result = predictor.predict(sample_dataframe)

            # Проверяем структуру результата
            assert isinstance(result, dict)
            assert "predictions" in result
            assert "is_defects" in result
            assert "defect_probabilities" in result
            assert "confidences" in result

            # Проверяем размеры
            n_samples = len(sample_dataframe)
            assert len(result["predictions"]) == n_samples
            assert len(result["is_defects"]) == n_samples
            assert len(result["defect_probabilities"]) == n_samples
            assert len(result["confidences"]) == n_samples

            # Проверяем типы данных
            assert all(isinstance(p, int) for p in result["predictions"])
            assert all(isinstance(d, bool) for d in result["is_defects"])
            assert all(isinstance(prob, float) for prob in result["defect_probabilities"])
            assert all(isinstance(conf, float) for conf in result["confidences"])

            print("✓ Тест 6 пройден: Базовое предсказание работает")

    def test_predict_batch(self, sample_dataframe, mock_model):
        """Тест 7: Пакетное предсказание"""
        # Мокаем загрузку модели
        with patch(
            "inference.predictor.DefectClassifier.safe_load_from_checkpoint"
        ) as mock_load:
            mock_load.return_value = MockModel()

            predictor = DefectPredictor("dummy.ckpt")
            predictor.scaler = None

            # Создаем список DataFrame
            data_list = [
                sample_dataframe.iloc[:3],
                sample_dataframe.iloc[3:7],
                sample_dataframe.iloc[7:],
            ]

            # Выполняем пакетное предсказание
            results = predictor.predict_batch(data_list)

            # Проверяем результат
            assert isinstance(results, list)
            assert len(results) == len(data_list)

            for i, result in enumerate(results):
                assert isinstance(result, dict)
                assert len(result["predictions"]) == len(data_list[i])

                print(f"  Батч {i}: {len(result['predictions'])} предсказаний")

            print("Тест 7 пройден: Пакетное предсказание работает")
