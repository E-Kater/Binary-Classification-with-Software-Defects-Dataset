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
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class ModelTrainer:
    """Класс для обучения модели"""

    def __init__(self, config: DictConfig):
        self.config = config
        setup_logging()

    def train(self):
        """Основной метод обучения"""

        # Настройка MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        with mlflow.start_run(run_name=self.config.experiment_name):
            # Логирование параметров
            mlflow.log_params(OmegaConf.to_container(self.config))

            # Инициализация DataModule
            datamodule = DefectDataModule(self.config)

            # Инициализация модели
            model = DefectClassifier(self.config)

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
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ]

            # Тренер
            trainer = pl.Trainer(
                max_epochs=self.config.training.max_epochs,
                callbacks=callbacks,
                accelerator=self.config.training.accelerator,
                devices=self.config.training.devices,
                log_every_n_steps=10,
                deterministic=True,
            )

            # Обучение
            logger.info("Начало обучения...")
            trainer.fit(model, datamodule=datamodule)

            # Тестирование
            logger.info("Тестирование модели...")
            trainer.test(model, datamodule=datamodule)

            # Сохранение модели в MLflow
            mlflow.pytorch.log_model(
                model, "model", registered_model_name="software_defect_classifier"
            )

            # Сохранение лучшей модели
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path:
                logger.info(f"Лучшая модель сохранена: {best_model_path}")
                mlflow.log_artifact(best_model_path)

            logger.info("Обучение завершено!")
