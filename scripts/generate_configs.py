import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_data_config(df: pd.DataFrame, output_path: Path):
    """Генерация конфигурации данных"""
    logger.info("Генерация конфигурации данных...")

    # Определяем числовые и категориальные признаки
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Убираем целевую переменную из признаков
    if "defects" in numeric_features:
        numeric_features.remove("defects")

    # Убираем ID если есть
    if "id" in numeric_features:
        numeric_features.remove("id")

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    config = {
        "data": {
            "raw_path": "data/raw/defects.csv",
            "processed_path": "data/processed/",
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
            "target_col": "defects",
        },
        "preprocessing": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "scaler": "standard",
        },
    }

    # Сохраняем конфигурацию
    with open(output_path / "data_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Конфигурация данных сохранена в {output_path / 'data_config.yaml'}")
    logger.info(f"Числовых признаков: {len(numeric_features)}")
    logger.info(f"Категориальных признаков: {len(categorical_features)}")

    return numeric_features


def generate_model_config(n_features: int, output_path: Path):
    """Генерация конфигурации модели"""
    logger.info("Генерация конфигурации модели...")

    # Вычисляем размеры скрытых слоев
    hidden_sizes = [
        min(128, max(64, n_features * 2)),
        min(64, max(32, n_features)),
        min(32, max(16, n_features // 2)),
    ]

    config = {
        "model": {
            "name": "DefectClassifier",
            "input_size": n_features,
            "hidden_sizes": hidden_sizes,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_classes": 2,
        }
    }

    # Сохраняем конфигурацию
    with open(output_path / "model_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Конфигурация модели сохранена в {output_path / 'model_config.yaml'}")
    logger.info(f"Архитектура: {n_features} -> {hidden_sizes} -> 2")


def generate_main_config(output_path: Path):
    """Генерация основной конфигурации"""
    logger.info("Генерация основной конфигурации...")

    config = {
        "defaults": ["data_config", "model_config", "training_config"],
        "project_name": "software_defect_prediction",
        "experiment_name": "baseline_experiment",
        "seed": 42,
        "hydra": {"run": {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"}},
    }

    # Сохраняем конфигурацию
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Основная конфигурация сохранена в {output_path / 'config.yaml'}")


def generate_training_config(output_path: Path):
    """Генерация конфигурации обучения"""
    logger.info("Генерация конфигурации обучения...")

    config = {
        "training": {
            "max_epochs": 50,
            "patience": 10,
            "monitor": "val_f1",
            "mode": "max",
            "accelerator": "cpu",
            "devices": 1,
        },
        "mlflow": {"tracking_uri": "mlruns", "experiment_name": "software_defects"},
    }

    # Сохраняем конфигурацию
    with open(output_path / "training_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(
        f"Конфигурация обучения сохранена в {output_path / 'training_config.yaml'}"
    )


def main():
    """Основная функция"""
    # Конфигурация
    DATA_PATH = Path("data/processed/train.csv")
    CONFIGS_DIR = Path("configs")

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Загружаем данные
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            logger.info(f"Данные загружены, размер: {df.shape}")

            # Генерируем конфигурации
            numeric_features = generate_data_config(df, CONFIGS_DIR)
            generate_model_config(len(numeric_features), CONFIGS_DIR)
            generate_main_config(CONFIGS_DIR)
            generate_training_config(CONFIGS_DIR)

            logger.info("\n=== Все конфигурации сгенерированы ===")

            # Сохраняем информацию о признаках
            features_info = {
                "total_features": len(numeric_features),
                "feature_names": numeric_features,
                "data_shape": df.shape,
                "target_distribution": df["defects"].value_counts().to_dict()
                if "defects" in df.columns
                else None,
            }

            with open(CONFIGS_DIR / "features_info.json", "w") as f:
                json.dump(features_info, f, indent=2)

            logger.info(
                f"Информация о признаках сохранена в {CONFIGS_DIR / 'features_info.json'}"  # noqa501
            )

        else:
            # Если данных нет, создаем конфигурации по умолчанию
            logger.warning(f"Файл данных не найден: {DATA_PATH}")
            logger.info("Создание конфигураций по умолчанию...")

            # Конфигурация по умолчанию на основе примера
            default_features = [
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

            config = {
                "data": {
                    "raw_path": "data/raw/defects.csv",
                    "processed_path": "data/processed/",
                    "test_size": 0.2,
                    "val_size": 0.1,
                    "random_state": 42,
                    "target_col": "defects",
                },
                "preprocessing": {
                    "numeric_features": default_features,
                    "categorical_features": [],
                    "scaler": "standard",
                },
            }

            with open(CONFIGS_DIR / "data_config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            generate_model_config(len(default_features), CONFIGS_DIR)
            generate_main_config(CONFIGS_DIR)
            generate_training_config(CONFIGS_DIR)

            logger.info("Конфигурации по умолчанию созданы")

    except Exception as e:
        logger.error(f"Ошибка при генерации конфигураций: {e}")
        raise


if __name__ == "__main__":
    main()
