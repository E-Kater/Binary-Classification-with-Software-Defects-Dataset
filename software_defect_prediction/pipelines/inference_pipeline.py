import json
import os
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from utils.logging import get_logger, setup_logging

# Добавляем путь к проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from inference.predictor import DefectPredictor  # noqa: E402

logger = get_logger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Пайплайн для инференса"""

    # Настройка логирования
    setup_logging()

    logger.info("=" * 60)
    logger.info("Software Defect Prediction - Инференс")
    logger.info("=" * 60)

    # Пути к модели и данным
    model_path = "models/best_model-v6.ckpt"
    scaler_path = "data/processed/scaler.joblib"
    test_data_path = "data/processed/test.csv"
    features_info_path = "data/processed/data_info.json"

    # Проверяем наличие модели
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена: {model_path}")
        logger.info("Сначала обучите модель: make train")
        return

    # Конвертируем конфигурацию Hydra в словарь
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Инициализируем предиктор
    logger.info("Инициализация предиктора...")
    predictor = DefectPredictor(model_path, config_dict)

    # Загружаем scaler
    if os.path.exists(scaler_path):
        predictor.load_scaler(scaler_path)

    # Загружаем имена признаков
    if os.path.exists(features_info_path):
        predictor.load_feature_names(features_info_path)

    # Вариант 1: Инференс на тестовых данных
    if os.path.exists(test_data_path):
        logger.info(f"\n1. Инференс на тестовых данных: {test_data_path}")

        # Оценка модели
        metrics = predictor.evaluate_on_test(test_data_path)

        if metrics:
            logger.info("\nМетрики модели:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric_name}: {value: .4f}")
                else:
                    logger.info(f"  {metric_name}: {value}")

            # Сохраняем метрики
            metrics_path = "inference_results/metrics.json"
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Метрики сохранены в {metrics_path}")

    # Вариант 2: Инференс на новых данных
    new_data_path = "data/raw/new_data.csv"
    if os.path.exists(new_data_path):
        logger.info(f"\n2. Инференс на новых данных: {new_data_path}")

        new_data = pd.read_csv(new_data_path)
        logger.info(f"Загружено {len(new_data)} новых образцов")

        # Предсказания
        result = predictor.predict(new_data)

        # Создаем DataFrame с результатами
        results_df = new_data.copy()
        results_df["prediction"] = result["predictions"]
        results_df["is_defect"] = results_df["prediction"] == 1

        if "defect_probability" in result:
            results_df["defect_probability"] = result["defect_probability"]
            results_df["confidence"] = np.max(result["probabilities"], axis=1)

        # Сохраняем результаты
        output_path = "inference_results/predictions.csv"
        results_df.to_csv(output_path, index=False)

        logger.info("\nРезультаты инференса: ")
        logger.info(f"  Всего образцов: {len(results_df)}")
        logger.info(f"  Предсказано дефектов: {results_df['is_defect'].sum()}")

        if "defect_probability" in results_df.columns:
            logger.info(
                f"  Средняя вероятность дефекта: {results_df['defect_probability'].mean(): .3f}"  # noqa501
            )

        logger.info(f"  Результаты сохранены в {output_path}")

        # Показываем несколько примеров
        logger.info("\nПримеры предсказаний (первые 5):")
        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            defect_status = "ДЕФЕКТ" if row["is_defect"] else "БЕЗ ДЕФЕКТА"
            prob = row.get("defect_probability", 0.0)
            logger.info(
                f"  Образец {i + 1}: {defect_status} (вероятность: {prob: .3f})"
            )  # noqa501

    # Вариант 3: Инференс на одиночных образцах (пример)
    logger.info("\n3. Пример инференса на одиночных образцах:")

    # Создаем несколько тестовых образцов
    sample_features = [
        {
            "loc": 22.0,
            "v(g)": 3.0,
            "ev(g)": 1.0,
            "iv(g)": 2.0,
            "n": 60.0,
            "v": 278.63,
            "l": 0.06,
            "d": 19.56,
            "i": 14.25,
            "e": 5448.79,
            "b": 0.09,
            "t": 302.71,
            "lOCode": 17,
            "lOComment": 1,
            "lOBlank": 1,
            "locCodeAndComment": 0,
            "uniq_Op": 16.0,
            "uniq_Opnd": 9.0,
            "total_Op": 38.0,
            "total_Opnd": 22.0,
            "branchCount": 5.0,
        },
        {
            "loc": 14.0,
            "v(g)": 2.0,
            "ev(g)": 1.0,
            "iv(g)": 2.0,
            "n": 32.0,
            "v": 151.27,
            "l": 0.14,
            "d": 7.0,
            "i": 21.11,
            "e": 936.71,
            "b": 0.05,
            "t": 52.04,
            "lOCode": 11,
            "lOComment": 0,
            "lOBlank": 1,
            "locCodeAndComment": 0,
            "uniq_Op": 11.0,
            "uniq_Opnd": 11.0,
            "total_Op": 18.0,
            "total_Opnd": 14.0,
            "branchCount": 3.0,
        },
    ]

    for i, features in enumerate(sample_features, 1):
        result = predictor.predict_single(features)
        status = "ДЕФЕКТ" if result["is_defect"] else "БЕЗ ДЕФЕКТА"
        logger.info(f"  Тестовый образец {i}: {status}")
        logger.info(f"    Вероятность дефекта: {result['defect_probability']: .3f}")
        logger.info(f"    Уверенность: {result['confidence']: .3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Инференс завершен!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
