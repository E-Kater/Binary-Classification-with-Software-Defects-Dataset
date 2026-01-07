import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path):
    """Загрузка данных из директории"""
    logger.info("Загрузка данных...")

    # Ищем CSV файлы
    csv_files = list(data_dir.glob("train.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV файлы не найдены в {data_dir}")

    # Загружаем первый найденный CSV
    data_path = csv_files[0]
    logger.info(f"Загрузка файла: {data_path.name}")

    df = pd.read_csv(data_path)
    logger.info(f"Данные загружены. Размер: {df.shape}")
    logger.info(f"Колонки: {df.columns.tolist()}")

    return df


def explore_data(df: pd.DataFrame):
    """Исследование данных"""
    logger.info("\n=== Исследование данных ===")

    # Основная информация
    logger.info("Основная информация:")
    logger.info(f"Размер данных: {df.shape}")
    logger.info(f"Типы данных: \n{df.dtypes}")

    # Пропущенные значения
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_info = pd.DataFrame(
        {"missing_count": missing, "missing_percentage": missing_pct}
    )

    logger.info("\nПропущенные значения:")
    logger.info(missing_info[missing_info["missing_count"] > 0])

    # Статистика по числовым колонкам
    logger.info("\nСтатистика по числовым колонкам:")
    logger.info(df.describe())

    # Распределение целевой переменной
    if "defects" in df.columns:
        target_dist = df["defects"].value_counts(normalize=True)
        logger.info("\nРаспределение целевой переменной: ")
        logger.info(f"TRUE: {target_dist.get(True, 0): .2%}")
        logger.info(f"FALSE: {target_dist.get(False, 0): .2%}")

    return missing_info


def prepare_features(df: pd.DataFrame):
    """Подготовка признаков"""
    logger.info("\n=== Подготовка признаков ===")

    # Сохраняем ID если есть
    id_column = None
    if "id" in df.columns:
        id_column = df["id"]
        df = df.drop(columns=["id"])

    # Проверяем целевую переменную
    target_column = "defects" if "defects" in df.columns else None

    if target_column:
        # Проверяем преобразование целевой переменной
        y = df["defects"].copy()
        y = y.map({"TRUE": 1, "FALSE": 0, True: 1, False: 0, "true": 1, "false": 0})

        nan_count = y.isna().sum()
        if nan_count > 0:
            print(f"Найдено {nan_count} NaN после преобразования")
            print(f"Проблемные значения: {df['defects'][y.isna()].unique()}")
        else:
            print("Преобразование целевой переменной успешно")

        # Отделяем признаки и целевую переменную
        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(f"Признаки: {X.shape}")
        logger.info(f"Целевая переменная: {y.shape}")

        return X, y, id_column
    else:
        # Для тестовых данных
        X = df.copy()
        logger.info(f"Тестовые данные. Признаки: {X.shape}")
        return X, None, id_column


def split_and_save_data(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """Разделение данных и сохранение"""
    logger.info("\n=== Разделение данных ===")

    # Создаем директории
    output_dir.mkdir(parents=True, exist_ok=True)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Разделение на train/validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_train,
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Масштабирование
    logger.info("\nМасштабирование данных...")
    scaler = StandardScaler()

    # Сохраняем имена колонок
    feature_names = X_train.columns.tolist()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Преобразуем обратно в DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Сохраняем данные
    logger.info("Сохранение данных...")

    # Сохраняем обработанные данные
    train_data = X_train_df.copy()
    train_data["defects"] = y_train.values
    train_data.to_csv(output_dir / "train.csv", index=False)

    val_data = X_val_df.copy()
    val_data["defects"] = y_val.values
    val_data.to_csv(output_dir / "val.csv", index=False)

    test_data = X_test_df.copy()
    test_data["defects"] = y_test.values
    test_data.to_csv(output_dir / "test.csv", index=False)

    # Сохраняем scaler
    joblib.dump(scaler, output_dir / "scaler.joblib")

    # Сохраняем метаданные
    metadata = {
        "feature_names": feature_names,
        "train_samples": len(X_train_df),
        "val_samples": len(X_val_df),
        "test_samples": len(X_test_df),
        "scaler": "StandardScaler",
        "test_size": test_size,
        "val_size": val_size,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Данные сохранены в {output_dir}")
    logger.info(f"Метаданные: {metadata}")

    return scaler, metadata


def save_raw_data(df: pd.DataFrame, output_dir: Path):
    """Сохранение сырых данных"""
    raw_dir = output_dir / "raw_processed"
    raw_dir.mkdir(exist_ok=True)

    # Сохраняем сырые данные
    raw_path = raw_dir / "raw_data.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Сырые данные сохранены в {raw_path}")

    # Сохраняем информацию о данных
    data_info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # в MB
    }

    info_path = raw_dir / "data_info.json"
    with open(info_path, "w") as f:
        json.dump(data_info, f, indent=2)

    logger.info(f"Информация о данных сохранена в {info_path}")


def main():
    """Основная функция"""
    # Конфигурация
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")

    try:
        # Загружаем данные
        df = load_data(RAW_DATA_DIR)

        # Сохраняем сырые данные
        save_raw_data(df, PROCESSED_DATA_DIR)

        # Исследуем данные
        explore_data(df)

        # Подготавливаем признаки
        X, y, ids = prepare_features(df)

        if y is not None:
            # Разделяем и сохраняем данные
            scaler, metadata = split_and_save_data(
                X, y, PROCESSED_DATA_DIR, test_size=0.2, val_size=0.1
            )

            logger.info("\n=== Препроцессинг завершен ===")
            logger.info(f"Общее количество образцов: {len(df)}")
            logger.info(f"Количество признаков: {X.shape[1]}")

            if ids is not None:
                logger.info(
                    f"ID колонка: {'сохранена' if ids is not None else 'отсутствует'}"
                )
        else:
            logger.warning(
                "Целевая переменная 'defects' не найдена. "
                "Скорее всего это тестовые данные."
            )

            # Сохраняем тестовые данные
            test_dir = PROCESSED_DATA_DIR / "test_data"
            test_dir.mkdir(parents=True, exist_ok=True)

            X.to_csv(test_dir / "test_features.csv", index=False)
            logger.info(f"Тестовые данные сохранены в {test_dir}")

    except Exception as e:
        logger.error(f"Ошибка при обработке данных: {e}")
        raise


if __name__ == "__main__":
    main()
