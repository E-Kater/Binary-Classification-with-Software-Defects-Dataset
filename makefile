.PHONY: help install setup test train preprocess clean lint download explore check generate-configs mlflow mlflow-ui mlflow-compare mlflow-serve mlflow-clean docs docs-clean docs-autogen docs-build docs-preview docs-open docs-serve
help:
	@echo "Доступные команды:"
	@echo "  install         Установка зависимостей"
	@echo "  setup           Настройка pre-commit"
	@echo "  test            Запуск тестов"
	@echo "  train           Обучение модели"
	@echo "  preprocess      Препроцессинг данных"
	@echo "  download        Скачивание данных с Kaggle"
	@echo "  explore         Анализ данных (EDA)"
	@echo "  check           Проверка установки"
	@echo "  generate-configs Генерация конфигураций"
	@echo "  clean           Очистка проекта"
	@echo "  lint            Проверка кода"
	@echo "  predict        - Запуск инференса (пайплайн)"
	@echo "  predict-cli    - Запуск инференса через CLI"
	@echo "  api            - Запуск FastAPI сервера"
	@echo "  serve          - Алиас для api"
	@echo "  check-inference - Проверка готовности инференса"
	@echo ""
	@echo "MLflow команды:"
	@echo "  mlflow          - Запуск MLflow UI (основная команда)"
	@echo "  mlflow-ui       - Запуск MLflow UI на порту 5000"
	@echo "  mlflow-serve    - Обслуживание модели через MLflow"
	@echo "  mlflow-compare  - Сравнение экспериментов"
	@echo "  mlflow-clean    - Очистка старых экспериментов"
	@echo "  mlflow-export   - Экспорт экспериментов в CSV"
	@echo ""
	@echo "Документация (Sphinx):"
	@echo "  docs-init       - Инициализация Sphinx"
	@echo "  docs-gen        - Генерация и сборка документации"
	@echo "  docs-view       - Просмотр документации в браузере"
	@echo "  docs-clean      - Очистка сгенерированной документации"
	@echo ""

# Пути для документации
DOCS_DIR = docs
SOURCE_DIR = $(DOCS_DIR)/source
BUILD_DIR = $(DOCS_DIR)/build
APIDOC_DIR = $(SOURCE_DIR)/api


install:
	poetry install

setup:
	pre-commit install

test:
	pytest tests/ -v --cov=src

train:
	python software-defect-prediction/pipelines/train_pipeline.py

preprocess:
	python scripts/data_preprocessing.py

download:
	python scripts/download_data.py

explore:
	python scripts/exploratory_analysis.py

check:
	python scripts/check_installation.py

generate-configs:
	python scripts/generate_configs.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache .hypothesis
	rm -rf **/__pycache__ **/.pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".DS_Store" -delete

lint:
	black software-defect-prediction/ tests/ scripts/
	isort software-defect-prediction/ tests/ scripts/
	flake8 software-defect-prediction/ tests/ scripts/
	mypy software-defect-prediction/

init-project: install setup generate-configs
	@echo "Проект инициализирован!"

predict:
	@echo "Запуск инференса..."
	python software-defect-prediction/pipelines/inference_pipeline.py

predict-cli:
	@echo "Запуск инференса через CLI..."
	python software-defect-prediction/inference/cli.py --data data/processed/test.csv --output predictions/predictions.csv

api:
	@echo "Запуск API сервера..."
	python software-defect-prediction/api/app.py

serve: api

check-inference:
	@echo "Проверка инференса..."
	@if [ -f "models/best_model.ckpt" ]; then \
		echo "✓ Модель найдена"; \
		python -c "import torch; print(f'PyTorch версия: {torch.__version__}')"; \
		python -c "from inference.predictor import DefectPredictor; print('✓ Модуль инференса импортирован')"; \
	else \
		echo "✗ Модель не найдена, сначала обучите модель: make train"; \
	fi

mlflow-ui:
	@echo "Starting MLflow UI..."
	poetry run python scripts/run_mlflow.py --port 5000 --host 0.0.0.0

mlflow-list:
	@echo "Listing MLflow experiments..."
	mlflow experiments search

dvc-push:
	@echo "Pushing data and model to DVC remote..."
	poetry run dvc push
	@echo "Pushed to DVC remote"

dvc-pull:
	@echo "Pulling data and model from DVC remote..."
	poetry run dvc pull
	@echo "Pulled from DVC remote"

dvc-status:
	@echo "Checking DVC status..."
	poetry run dvc status
# Документация Sphinx
docs-init:
	@echo "Инициализация Sphinx..."
	sphinx-quickstart docs --sep --project "Project" --author "You" --extensions "sphinx.ext.autodoc"

docs-gen:
	@echo "Генерация документации..."
	sphinx-apidoc -o docs/source . -f
	sphinx-build -b html docs/source docs/build/html

docs-view:
	@echo "Открытие документации..."
	open docs/build/html/index.html 2>/dev/null || xdg-open docs/build/html/index.html 2>/dev/null || echo "Откройте: docs/build/html/index.html"

docs-clean:
	@echo "Очистка документации..."
	rm -rf docs/build docs/source/*.rst
