import logging
import os
import zipfile
from pathlib import Path

import kaggle
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_kaggle_api():
    """Настройка Kaggle API"""
    # Проверяем наличие kaggle.json
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        # Проверяем переменные окружения
        load_dotenv()
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")

        if kaggle_username and kaggle_key:
            # Создаем kaggle.json из переменных окружения
            kaggle_dir.mkdir(exist_ok=True)
            kaggle_json.write_text(
                f'{{"username": "{kaggle_username}", "key": "{kaggle_key}"}}'
            )
            kaggle_json.chmod(0o600)
            logger.info("Kaggle API настроен через переменные окружения")
        else:
            raise Exception(
                "Не найден kaggle.json и не установлены переменные окружения "
                "KAGGLE_USERNAME и KAGGLE_KEY"
            )
    else:
        logger.info("Kaggle API уже настроен")


def download_competition_data(competition_name: str, output_dir: Path):
    """
    Скачивание данных конкурса Kaggle

    Args:
        competition_name: Имя конкурса (например, 'playground-series-s3e23')
        output_dir: Директория для сохранения данных
    """
    try:
        logger.info(f"Скачивание данных конкурса: {competition_name}")

        # Создаем директории если их нет
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        # Скачиваем данные
        kaggle.api.competition_download_files(
            competition_name, path=raw_dir, quiet=False
        )  # noqa501

        # Распаковываем архив
        zip_files = list(raw_dir.glob("*.zip"))
        if zip_files:
            for zip_file in zip_files:
                logger.info(f"Распаковка {zip_file.name}")
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(raw_dir)

                # Удаляем архив после распаковки
                zip_file.unlink()
                logger.info(f"Архив {zip_file.name} удален")

        logger.info(f"Данные успешно скачаны в {raw_dir}")

        # Показываем скачанные файлы
        data_files = list(raw_dir.glob("*"))
        logger.info(f"Скачанные файлы: {[f.name for f in data_files]}")

        return raw_dir

    except Exception as e:
        logger.error(f"Ошибка при скачивании данных: {e}")
        raise


def main():
    """Основная функция"""
    # Настройка Kaggle API
    setup_kaggle_api()

    # Конфигурация
    COMPETITION_NAME = "playground-series-s3e23"
    DATA_DIR = Path("data")

    # Скачиваем данные
    download_competition_data(COMPETITION_NAME, DATA_DIR)


if __name__ == "__main__":
    main()
