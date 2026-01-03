#!/usr/bin/env python3
"""
Командный интерфейс для инференса
"""

import argparse
import json
import os
import sys

import pandas as pd

# Добавляем путь к проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from inference.predictor import DefectPredictor  # noqa402


def main():
    parser = argparse.ArgumentParser(
        description="Инференс модели предсказания дефектов ПО"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model-v6.ckpt",
        help="Путь к обученной модели",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="data/processed/scaler.joblib",
        help="Путь к scaler",  # noqa501
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Путь к данным для предсказания (CSV файл)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Путь для сохранения результатов",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Путь к конфигурационному файлу",
    )

    args = parser.parse_args()

    # Проверяем файлы
    if not os.path.exists(args.model):
        print(f"Ошибка: Модель не найдена: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"Ошибка: Данные не найдены: {args.data}")
        sys.exit(1)

    # Загружаем конфигурацию если есть
    config = None
    if os.path.exists(args.config):
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Инициализируем предиктор
    print("Загрузка модели...")
    predictor = DefectPredictor(args.model, config)

    if os.path.exists(args.scaler):
        predictor.load_scaler(args.scaler)

    # Загружаем данные
    print(f"Загрузка данных из {args.data}...")
    data = pd.read_csv(args.data)
    print(f"Загружено {len(data)} образцов")

    # Предсказание
    print("Выполнение предсказаний...")
    result = predictor.predict(data)

    # Создаем DataFrame с результатами
    results_df = data.copy()
    results_df["prediction"] = result["predictions"]
    results_df["is_defect"] = results_df["prediction"] == 1

    if "defect_probability" in result:
        results_df["defect_probability"] = result["defect_probability"]
        results_df["confidence"] = [max(probs) for probs in result["probabilities"]]

    # Сохраняем результаты
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(args.output, index=False)

    # Выводим статистику
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ИНФЕРЕНСА")
    print("=" * 60)
    print(f"Всего образцов: {len(results_df)}")
    print(f"Предсказано дефектов: {results_df['is_defect'].sum()}")
    print(f"Процент дефектов: {results_df['is_defect'].mean(): .2%}")

    if "defect_probability" in results_df.columns:
        print(
            f"Средняя вероятность дефекта: {results_df['defect_probability'].mean(): .3f}"  # noqa501
        )
        print(
            f"Максимальная вероятность: {results_df['defect_probability'].max(): .3f}"
        )  # noqa501
        print(
            f"Минимальная вероятность: {results_df['defect_probability'].min(): .3f}"
        )  # noqa501

    print(f"\nРезультаты сохранены в: {args.output}")

    # Сохраняем сводную статистику
    stats = {
        "total_samples": len(results_df),
        "defect_count": int(results_df["is_defect"].sum()),
        "defect_percentage": float(results_df["is_defect"].mean()),
        "model_path": args.model,
        "data_path": args.data,
    }

    if "defect_probability" in results_df.columns:
        stats.update(
            {
                "avg_defect_probability": float(
                    results_df["defect_probability"].mean()
                ),  # noqa501
                "max_defect_probability": float(
                    results_df["defect_probability"].max()
                ),  # noqa501
                "min_defect_probability": float(
                    results_df["defect_probability"].min()
                ),  # noqa501
            }
        )

    stats_path = os.path.join(
        os.path.dirname(args.output), "inference_stats.json"
    )  # noqa501
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Статистика сохранена в: {stats_path}")


if __name__ == "__main__":
    main()
