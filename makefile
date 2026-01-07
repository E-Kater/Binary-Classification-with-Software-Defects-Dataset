.PHONY: help install setup test train preprocess clean lint download explore check mlflow-ui mlflow-list docs-init docs-gen  docs-view docs-clean  triton-convert convert-onnx triton-start triton-stop pipeline-full pipeline-data pipeline-train pipeline-deploy logs
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
	@echo "  clean           Очистка проекта"
	@echo "  lint            Проверка кода"
	@echo "  predict        - Запуск инференса (пайплайн)"
	@echo "  api            - Запуск FastAPI сервера"
	@echo "  serve          - Алиас для api"
	@echo "  check-inference - Проверка готовности инференса"
	@echo "  logs            Просмотр логов (logs/app.log)"
	@echo ""
	@echo "ПАЙПЛАЙНЫ:"
	@echo "  pipeline-full   Полный пайплайн (данные → обучение → деплой)"
	@echo "  pipeline-data   Пайплайн данных (скачивание → обработка → EDA)"
	@echo "  pipeline-train  Пайплайн обучения (препроцессинг → обучение → оценка)"
	@echo "  pipeline-deploy Пайплайн деплоя (конвертация → сервинг → тестирование)"
	@echo ""
	@echo "MLflow команды:"
	@echo "  mlflow-ui       - Запуск MLflow UI на порту 5000"
	@echo "  mlflow-list   - Список MLflow экспериментов"
	@echo ""
	@echo "Документация (Sphinx):"
	@echo "  docs-init       - Инициализация Sphinx"
	@echo "  docs-gen        - Генерация и сборка документации"
	@echo "  docs-view       - Просмотр документации в браузере"
	@echo "  docs-clean      - Очистка сгенерированной документации"
	@echo ""
	@echo "ONNX команды:"
	@echo "  convert-onnx    - Конвертация модели в ONNX"
	@echo ""
	@echo "TensorRT команды:"
	@echo "  convert-tensorrt    - Конвертация в TensorRT"
	@echo ""
	@echo "Triton команды:"
	@echo "  triton-convert    - Конвертация модели для Triton"
	@echo "  triton-start      - Запуск Triton Inference Server"
	@echo "  triton-stop       - Остановка Triton"


# Пути для документации
DOCS_DIR = docs
SOURCE_DIR = $(DOCS_DIR)/source
BUILD_DIR = $(DOCS_DIR)/build
APIDOC_DIR = $(SOURCE_DIR)/api

# Triton Inference Server
MODEL_NAME = defect_classifier
TRITON_IMAGE = nvcr.io/nvidia/tritonserver:23.10-py3


install:
	poetry install

setup:
	pre-commit install

test:
	pytest tests/ -v

train:
	python software_defect_prediction/pipelines/train_pipeline.py

preprocess:
	python scripts/data_preprocessing.py

download:
	python scripts/download_data.py

explore:
	python scripts/exploratory_analysis.py

check:
	python scripts/check_installation.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache .hypothesis
	rm -rf **/__pycache__ **/.pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".DS_Store" -delete

lint:
	pre-commit run --all-files

predict:
	@echo "Запуск инференса..."
	python software-defect-prediction/pipelines/inference_pipeline.py

predict-cli:
	@echo "Запуск инференса через CLI..."
	python software-defect-prediction/inference/cli.py --data data/processed/test.csv --output predictions/predictions.csv

api:
	@echo "Запуск API сервера..."
	python software_defect_prediction/api/app.py

serve: api

check-inference:
	@echo "Проверка инференса..."
	@if [ -f "models/best_model.ckpt" ]; then \
		echo "✓ Модель найдена"; \
		python -c "import torch; print(f'PyTorch версия: {torch.__version__}')"; \
		python -c "from software_defect_prediction.inference.predictor import DefectPredictor; print('✓ Модуль инференса импортирован')"; \
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
	poetry run dvc pull --force
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

triton-convert:
	@echo "Конвертация модели для Triton..."
	python scripts/convert_to_triton.py

triton-start:
	@echo "Запуск Triton Inference Server..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		cd docker/triton && docker-compose up -d triton; \
		echo "Triton запущен на http://localhost:8000"; \
		echo "Метрики: http://localhost:8002/metrics"; \
	else \
		echo "Docker Compose не установлен"; \
	fi

triton-stop:
	@echo "Остановка Triton..."
	@cd docker/triton && docker-compose down

convert-onnx:
	python scripts/convert_to_onnx.py
convert-tensorrt:
	@echo "Converting to TensorRT..."
	python scripts/convert_to_tensorrt.py
# Полный пайплайн от данных до деплоя
pipeline-full: pipeline-data pipeline-train pipeline-deploy
	@echo "=" * 60
	@echo "ПОЛНЫЙ ПАЙПЛАЙН ЗАВЕРШЕН!"
	@echo "=" * 60
	@echo "Результаты:"
	@echo "  Данные загружены и обработаны"
	@echo "  Модель обучена и оценена"
	@echo "  Модель конвертирована и развернута"
	@echo "  API сервер запущен"
	@echo "=" * 60

# Пайплайн данных (скачивание → обработка → анализ)
pipeline-data:
	@echo "=" * 60
	@echo "ЗАПУСК ПАЙПЛАЙНА ДАННЫХ"
	@echo "=" * 60
	@echo "1. Скачивание данных..."
	@$(MAKE) download
	@echo "Данные скачаны"
	@echo ""
	@echo "2. Препроцессинг данных..."
	@$(MAKE) preprocess
	@echo "Данные обработаны"
	@echo ""
	@echo "3. Анализ данных (EDA)..."
	@$(MAKE) explore
	@echo "EDA завершен"
	@echo "ПАЙПЛАЙН ДАННЫХ ЗАВЕРШЕН!"
	@echo "=" * 60

# Пайплайн обучения (препроцессинг → обучение → оценка)
pipeline-train:
	@echo "=" * 60
	@echo "ЗАПУСК ПАЙПЛАЙНА ОБУЧЕНИЯ"
	@echo "=" * 60
	@echo "1. Выгрузка из DVC..."
	@$(MAKE) dvc-pull
	@echo "✓ Данные выгружены тз DVC"
	@echo "2. Проверка данных..."
	@if [ ! -f "data/processed/train.csv" ]; then \
		echo "✗ Данные не найдены, запустите: make pipeline-data"; \
		exit 1; \
	fi
	@echo "Данные найдены"
	@echo ""
	@echo "3. Запуск обучения модели..."
	@$(MAKE) train
	@echo "Модель обучена"
	@echo ""
	@echo "4. Сохранение в DVC..."
	@$(MAKE) dvc-push
	@echo "✓ Результаты сохранены в DVC"
	@echo "=" * 60
	@echo "ПАЙПЛАЙН ОБУЧЕНИЯ ЗАВЕРШЕН!"
	@echo "=" * 60

# Пайплайн деплоя (конвертация → сервинг → тестирование)
pipeline-deploy:
	@echo "=" * 60
	@echo "ЗАПУСК ПАЙПЛАЙНА ДЕПЛОЯ"
	@echo "=" * 60
	@echo "1. Конвертация в ONNX..."
	@$(MAKE) convert-onnx
	@echo "✓ Модель конвертирована в ONNX"
	@echo ""
	@echo "2. Генерация документации"
	@$(MAKE) docs-gen
	@echo "✓ Документация сгенерирована"
	@echo ""
	@echo "3. Запуск API сервера..."
	@$(MAKE) api &
	@sleep 5  # Даем время серверу запуститься
	@echo "✓ API сервер запущен"
	@echo ""
	@echo "4. Мониторинг через MLflow..."
	@echo "   Запустите отдельно: make mlflow-ui"
	@echo "=" * 60
	@echo "ПАЙПЛАЙН ДЕПЛОЯ ЗАВЕРШЕН!"
	@echo "=" * 60
	@echo "Сервер запущен на http://localhost:8000"
	@echo "Документация API: http://localhost:8000/docs"
logs:
	@echo "Просмотр логов из папки logs/app.log"
	@if [ -f "logs/app.log" ]; then \
		tail -f logs/app.log; \
	else \
		echo "Файл логов не найден: logs/app.log"; \
		echo "Создайте директорию logs или убедитесь в наличии файла"; \
	fi
