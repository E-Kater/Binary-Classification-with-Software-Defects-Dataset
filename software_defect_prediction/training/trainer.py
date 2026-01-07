from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from data.datamodule import DefectDataModule
from models.model import DefectClassifier
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class ModelTrainer:
    """Класс для обучения модели"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.reports_dir = Path("reports/figures")
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.val_f1_history: List[float] = []
        self.train_f1_history: List[float] = []
        self.epochs_history: List[float] = []
        setup_logging()

    def train(self):
        """Основной метод обучения"""
        # Настройка MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with mlflow.start_run(run_name=self.config.mlflow.experiment_name):
            # Логирование параметров
            mlflow.log_params(OmegaConf.to_container(self.config))

            # Инициализация DataModule
            datamodule = DefectDataModule(self.config)

            # Инициализация модели
            model = DefectClassifier(self.config)

            # Добавляем TensorBoard logger
            tensorboard_logger = TensorBoardLogger(
                save_dir="lightning_logs",
                name=self.config.mlflow.experiment_name,
                version=timestamp,
            )
            metrics_callback = MetricsCollectorCallback(self)
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor=self.config.training.monitor,
                    patience=self.config.training.patience,
                    mode=self.config.training.mode,
                    verbose=True,
                ),
                ModelCheckpoint(
                    monitor="val_f1",
                    mode="max",
                    save_top_k=1,
                    dirpath="models/",
                    filename="best_model",
                    save_weights_only=True,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                metrics_callback,
            ]

            # Тренер
            trainer = pl.Trainer(
                max_epochs=self.config.training.max_epochs,
                callbacks=callbacks,
                accelerator=self.config.training.accelerator,
                devices=self.config.training.devices,
                log_every_n_steps=1,
                deterministic=True,
                logger=[tensorboard_logger],
                enable_progress_bar=True,
                enable_model_summary=True,
            )

            # Обучение
            logger.info("Начало обучения...")
            trainer.fit(model, datamodule=datamodule)

            # Тестирование
            logger.info("Тестирование модели...")
            test_results = trainer.test(model, datamodule=datamodule)

            for metric, value in test_results[0].items():
                mlflow.log_metric(f"test_{metric}", value)

            # Сохранение модели в MLflow
            mlflow.pytorch.log_model(
                model, "model", registered_model_name="software_defect_classifier"
            )

            # Сохранение лучшей модели
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path:
                logger.info(f"Лучшая модель сохранена: {best_model_path}")
                mlflow.log_artifact(best_model_path)
                OmegaConf.update(
                    self.config, "model.model_path", best_model_path, merge=True
                )
                # Сохраняем
                OmegaConf.save({"model": self.config.model}, "configs/model_config.yaml")

            logger.info("Обучение завершено!")
            self._export_tensorboard_figures(tensorboard_logger, timestamp)

    def _export_tensorboard_figures(self, logger, timestamp):
        """Экспортирует графики из TensorBoard"""
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            log_dir = Path(logger.log_dir)
            event_file = next(log_dir.glob("events.out.tfevents.*"))

            # Загружаем события
            event_acc = EventAccumulator(str(event_file))
            event_acc.Reload()

            # Получаем скаляры
            tags = event_acc.Tags()["scalars"]

            for tag in tags:
                if "loss" in tag or "f1" in tag:
                    events = event_acc.Scalars(tag)
                    epochs = [e.step for e in events]
                    values = [e.value for e in events]

                    # Создаем график
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(epochs, values, linewidth=2)
                    ax.set_xlabel("Step")
                    ax.set_ylabel(tag.replace("_", " ").title())
                    ax.set_title(f"{tag} over step")
                    ax.grid(True, alpha=0.3)

                    # Сохраняем
                    fig_path = self.reports_dir / f"{tag}_{timestamp}.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    # Логируем в MLflow
                    mlflow.log_artifact(str(fig_path), "tensorboard_figures")

        except Exception as e:
            logger.warning(f"Ошибка при экспорте из TensorBoard: {e}")


class MetricsCollectorCallback(pl.Callback):
    """Callback для сбора метрик во время обучения"""

    def __init__(self, trainer: ModelTrainer):
        super().__init__()
        self.trainer = trainer

    def on_train_epoch_end(self, trainer, pl_module):
        """Собирает метрики в конце каждой эпохи"""
        try:
            # Получаем метрики из логгера
            metrics = trainer.callback_metrics
            logger.info(f"DEBUG Epoch {trainer.current_epoch} metrics: {metrics}")
            # Сохраняем loss
            train_loss = metrics.get("train_loss")
            val_loss = metrics.get("val_loss")
            val_f1 = metrics.get("val_f1")
            train_f1 = metrics.get("train_f1")

            current_epoch = trainer.current_epoch

            if train_loss is not None:
                self.trainer.train_loss_history.append(train_loss.item())

            if val_loss is not None:
                self.trainer.val_loss_history.append(val_loss.item())

            if val_f1 is not None:
                self.trainer.val_f1_history.append(val_f1.item())

            if train_f1 is not None:
                self.trainer.train_f1_history.append(train_f1.item())

            self.trainer.epochs_history.append(current_epoch)

            # Логируем в MLflow
            if mlflow.active_run():
                if train_loss is not None:
                    mlflow.log_metric("train_loss", train_loss.item(), step=current_epoch)
                if val_loss is not None:
                    mlflow.log_metric("val_loss", val_loss.item(), step=current_epoch)
                if val_f1 is not None:
                    mlflow.log_metric("val_f1", val_f1.item(), step=current_epoch)
                if val_f1 is not None:
                    mlflow.log_metric("train_f1", train_f1.item(), step=current_epoch)

        except Exception as e:
            logger.warning(f"Ошибка при сборе метрик: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Собираем метрики после каждого батча"""
        try:
            # Собираем метрики из outputs если они есть
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if key.endswith("_loss") or key.endswith("_f1"):
                        if key not in self.current_epoch_metrics:
                            self.current_epoch_metrics[key] = []
                        self.current_epoch_metrics[key].append(float(value))
        except Exception as e:
            logger.info(e)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Обработка метрик в конце валидационной эпохи"""
        try:
            metrics = trainer.callback_metrics
            logger.info("metrics ", metrics)

            current_epoch = trainer.current_epoch

            # Val loss
            if "val_loss" in metrics:
                val_loss = metrics["val_loss"].item()
                if len(self.trainer.val_loss_history) <= current_epoch:
                    self.trainer.val_loss_history.append(val_loss)
                mlflow.log_metric("val_loss", val_loss, step=current_epoch)

            if "val_f1" in metrics:
                val_f1 = metrics["val_f1"].item()
                if len(self.trainer.val_f1_history) <= current_epoch:
                    self.trainer.val_f1_history.append(val_f1)
                mlflow.log_metric("val_f1", val_f1, step=current_epoch)

            if "train_f1" in metrics:
                train_f1 = metrics["train_f1"].item()
                if len(self.trainer.train_f1_history) <= current_epoch:
                    self.trainer.train_f1_history.append(train_f1)
                mlflow.log_metric("train_f1", train_f1, step=current_epoch)

        except Exception as e:
            logger.warning(f"Ошибка при сборе валидационных метрик: {e}")
