import importlib
import logging
import subprocess
import sys
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Проверка версии Python"""
    logger.info("Проверка версии Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(
            f"✗ Требуется Python 3.9+, текущая версия: {version.major}.{version.minor}"
        )
        return False


def check_package(package_name, import_name=None):
    """Проверка наличия пакета"""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        logger.info(f"✓ {package_name}")
        return True
    except ImportError:
        logger.error(f"✗ {package_name} не установлен")
        return False


def check_poetry():
    """Проверка Poetry"""
    logger.info("Проверка Poetry...")
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ Poetry установлен: {result.stdout.strip()}")
            return True
        else:
            logger.error("✗ Poetry не установлен или не настроен")
            return False
    except FileNotFoundError:
        logger.error("✗ Poetry не найден в PATH")
        return False


def check_git():
    """Проверка Git"""
    logger.info("Проверка Git...")
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ Git установлен: {result.stdout.strip()}")
            return True
        else:
            logger.error("✗ Git не установлен")
            return False
    except FileNotFoundError:
        logger.error("✗ Git не найден в PATH")
        return False


def check_dvc():
    """Проверка DVC"""
    logger.info("Проверка DVC...")
    try:
        result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ DVC установлен: {result.stdout.strip()}")
            return True
        else:
            logger.error("✗ DVC не установлен")
            return False
    except FileNotFoundError:
        logger.error("✗ DVC не найден в PATH")
        return False


def check_directory_structure():
    """Проверка структуры директорий"""
    logger.info("Проверка структуры директорий...")

    required_dirs = [
        "data/raw",
        "data/processed",
        "software_defect_prediction",
        "tests",
        "models",
        "notebooks",
        "scripts",
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        logger.error(f"✗ Отсутствуют директории: {missing_dirs}")
        return False
    else:
        logger.info("✓ Структура директорий в порядке")
        return True


def check_config_files():
    """Проверка конфигурационных файлов"""
    logger.info("Проверка конфигурационных файлов...")

    required_files = [
        "pyproject.toml",
        ".pre-commit-config.yaml",
        ".gitignore",
        "Makefile",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error(f"✗ Отсутствуют файлы: {missing_files}")
        return False
    else:
        logger.info("✓ Конфигурационные файлы в порядке")
        return True


def main():
    """Основная функция проверки"""
    logger.info("=" * 60)
    logger.info("Проверка установки проекта Software Defect Prediction")
    logger.info("=" * 60)

    checks = []

    # Проверяем Python
    checks.append(("Python версия", check_python_version()))

    # Проверяем инструменты
    checks.append(("Poetry", check_poetry()))
    checks.append(("Git", check_git()))
    checks.append(("DVC", check_dvc()))

    # Проверяем пакеты Python
    logger.info("\nПроверка Python пакетов...")
    packages = [
        ("torch", "torch"),
        ("pytorch-lightning", "pytorch_lightning"),
        ("hydra", "hydra"),
        ("mlflow", "mlflow"),
        ("pandas", "pandas"),
        ("loguru", "loguru"),
        ("pre-commit", "pre_commit"),
    ]

    for package_name, import_name in packages:
        checks.append((package_name, check_package(package_name, import_name)))

    # Проверяем структуру проекта
    checks.append(("Структура директорий", check_directory_structure()))
    checks.append(("Конфигурационные файлы", check_config_files()))

    # Вывод итогов
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГИ ПРОВЕРКИ:")
    logger.info("=" * 60)

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    for name, status in checks:
        status_symbol = "✓" if status else "✗"
        logger.info(f"{status_symbol} {name}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Пройдено: {passed}/{total}")

    if passed == total:
        logger.info("✓ Все проверки пройдены успешно!")
        logger.info("\nПроект готов к работе. Следующие шаги:")
        logger.info("1. Установите pre-commit: make setup")
        logger.info("2. Обработайте данные: make preprocess")
        logger.info("3. Запустите обучение: make train")
    else:
        logger.error("✗ Некоторые проверки не пройдены")
        logger.info("\nДля исправления проблем:")
        logger.info("1. Установите недостающие пакеты: poetry install")
        logger.info("2. Создайте недостающие директории")
        logger.info("3. Установите недостающие инструменты")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
