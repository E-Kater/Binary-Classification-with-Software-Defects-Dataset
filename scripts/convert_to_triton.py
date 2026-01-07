#!/usr/bin/env python3
"""
Конвертация модели PyTorch в формат Triton
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import DefectClassifier


def convert_pytorch_to_triton():
    """Конвертация модели PyTorch в формат Triton"""

    print("=" * 60)
    print("Конвертация модели для Triton Inference Server")
    print("=" * 60)

    # Пути
    model_checkpoint = "models/best_model.ckpt"
    triton_model_dir = "triton_models/model_repository/software_defects/1"

    # Создаем директории
    os.makedirs(triton_model_dir, exist_ok=True)

    # Загружаем модель
    if not os.path.exists(model_checkpoint):
        print(f"Ошибка: Модель не найдена: {model_checkpoint}")
        print("Сначала обучите модель: make train")
        return False

    try:
        # Загружаем модель PyTorch Lightning
        model = DefectClassifier.load_from_checkpoint(model_checkpoint)
        model.eval()

        # Создаем пример ввода для трассировки
        example_input = torch.randn(1, 21)

        # Трассируем модель
        traced_model = torch.jit.trace(model, example_input)

        # Сохраняем в формате TorchScript
        model_path = os.path.join(triton_model_dir, "model.pt")
        traced_model.save(model_path)

        print(f"Модель сохранена: {model_path}")

        # Сохраняем метаданные
        metadata = {
            "input_shape": [21],
            "output_shape": [2],
            "model_type": "software_defects",
            "framework": "pytorch",
            "version": "1",
        }

        import json

        metadata_path = os.path.join(triton_model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Метаданные сохранены: {metadata_path}")

        # Проверяем модель
        print("\nПроверка конвертированной модели...")
        loaded_model = torch.jit.load(model_path)

        # Тестовый inference
        test_input = torch.randn(2, 21)  # batch_size=2
        with torch.no_grad():
            output = loaded_model(test_input)

        print("  Тестовый inference успешен")
        print(f"  Входная форма: {test_input.shape}")
        print(f"  Выходная форма: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ Ошибка при конвертации: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_ensemble_model():
    """Создание ensemble модели для препроцессинга"""

    print("\n" + "=" * 60)
    print("Создание ensemble модели")
    print("=" * 60)

    ensemble_dir = "triton_models/model_repository/software_defects_ensemble"

    # 1. Модель для препроцессинга
    preprocess_dir = os.path.join(ensemble_dir, "preprocess", "1")
    os.makedirs(preprocess_dir, exist_ok=True)

    # Создаем простую модель препроцессинга на Python
    preprocess_code = """
import numpy as np
import json
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # Загружаем scaler если есть
        try:
            import joblib
            self.scaler = joblib.load("/models/scaler.joblib")
        except:
            self.scaler = None

        print("Preprocess model initialized")

    def execute(self, requests):
        responses = []

        for request in requests:
            # Получаем входные данные
            input_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_FEATURES")
            raw_features = input_tensor.as_numpy()

            # Препроцессинг
            if self.scaler is not None:
                processed = self.scaler.transform(raw_features)
            else:
                processed = raw_features.astype(np.float32)

            # Создаем выходной тензор
            output_tensor = pb_utils.Tensor("PROCESSED_FEATURES", processed)

            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def finalize(self):
        print("Preprocess model finalized")
"""

    preprocess_path = os.path.join(preprocess_dir, "model.py")
    with open(preprocess_path, "w") as f:
        f.write(preprocess_code)

    # Конфигурация препроцессинга
    preprocess_config = """
name: "preprocess"
backend: "python"
max_batch_size: 32
input [
  {
    name: "RAW_FEATURES"
    data_type: TYPE_FP32
    dims: [21]
  }
]
output [
  {
    name: "PROCESSED_FEATURES"
    data_type: TYPE_FP32
    dims: [21]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
"""

    config_path = os.path.join(os.path.dirname(preprocess_dir), "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(preprocess_config)

    print(f"✓ Модель препроцессинга создана: {preprocess_dir}")

    # 2. Ensemble конфигурация
    ensemble_config_dir = os.path.join(ensemble_dir, "1")
    os.makedirs(ensemble_config_dir, exist_ok=True)

    ensemble_config = """
name: "software_defects_ensemble"
platform: "ensemble"
max_batch_size: 32
input [
  {
    name: "RAW_FEATURES"
    data_type: TYPE_FP32
    dims: [21]
  }
]
output [
  {
    name: "PREDICTIONS"
    data_type: TYPE_FP32
    dims: [2]
  },
  {
    name: "PROBABILITIES"
    data_type: TYPE_FP32
    dims: [2]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "preprocess"
      model_version: -1
      input_map {
        key: "RAW_FEATURES"
        value: "RAW_FEATURES"
      }
      output_map {
        key: "PROCESSED_FEATURES"
        value: "processed_features"
      }
    },
    {
      model_name: "software_defects"
      model_version: -1
      input_map {
        key: "input__0"
        value: "processed_features"
      }
      output_map {
        key: "output__0"
        value: "PREDICTIONS"
      }
    }
  ]
}
"""

    ensemble_config_path = os.path.join(ensemble_config_dir, "config.pbtxt")
    with open(ensemble_config_path, "w") as f:
        f.write(ensemble_config)

    print(f"✓ Ensemble модель создана: {ensemble_config_dir}")
    return True


if __name__ == "__main__":
    success = convert_pytorch_to_triton()

    if success:
        # Создаем ensemble модель
        create_ensemble_model()

        print("\n" + "=" * 60)
        print("✅ Конвертация завершена!")
        print("=" * 60)
        print("\nСтруктура Triton модели:")
        print("  triton_models/model_repository/")
        print("  ├── software_defects/     # Основная модель")
        print("  └── software_defects_ensemble/  # Ensemble с препроцессингом")
        print("\nЗапустите Triton:")
        print("  make triton_models-start")
    else:
        print("\nКонвертация не удалась")
