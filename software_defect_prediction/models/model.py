import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from utils.logging import get_logger

from .metric import calculate_metrics

logger = get_logger(__name__)


class DefectClassifier(pl.LightningModule):
    """Модель для классификации дефектов ПО"""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        model_config = config.model

        # Создание слоев
        layers = []
        input_size = model_config.input_size

        for hidden_size in model_config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_config.dropout_rate))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, model_config.num_classes))

        self.network = nn.Sequential(*layers)

        self.loss_fn = None
        self.class_weights = None
        # Для метрик
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.network(x)

    def setup(self, stage=None):
        """Вызывается после инициализации DataLoader"""
        if stage == "fit" and self.class_weights is None:
            # Собираем все метки из train dataloader
            logger.info("Собираем данные для расчета весов классов...")

            # Получаем train dataloader
            train_dataloader = self.trainer.datamodule.train_dataloader()

            all_labels = []
            for batch in train_dataloader:
                _, y = batch
                all_labels.extend(y.cpu().numpy())

            # Рассчитываем веса
            self._calculate_class_weights(all_labels)

    def _calculate_class_weights(self, labels):
        """Рассчитывает веса классов"""
        labels = np.array(labels)
        classes = np.arange(self.config.model.num_classes)

        # Проверяем, все ли классы присутствуют
        present_classes = np.unique(labels)
        if len(present_classes) < len(classes):
            logger.warning(
                f"Некоторые классы отсутствуют в данных: "
                f"присутствуют {present_classes}, ожидались {classes}"
            )

        # Рассчитываем веса
        try:
            class_weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=labels
            )
        except ValueError:
            # Если какие-то классы отсутствуют, используем равные веса
            logger.warning("Используем равные веса классов")
            class_weights = np.ones(len(classes))
        # Преобразуем в тензор
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Создаем функцию потерь
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Логируем результат
        logger.info(f"Веса классов: {self.class_weights.tolist()}")
        logger.info(
            f"Распределение классов: {np.bincount(labels.astype(int), minlength=len(classes)).tolist()}"  # noqa501
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Если веса еще не рассчитаны, рассчитываем на первом батче
        if self.loss_fn is None:
            self._calculate_class_weights(y.cpu().numpy())

        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_preds.extend(preds.cpu().numpy())
        self.train_targets.extend(y.cpu().numpy())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.train_preds:
            metrics = calculate_metrics(
                np.array(self.train_targets), np.array(self.train_preds)
            )
            for name, value in metrics.items():
                self.log(f"train_{name}", value, prog_bar=True)

            self.train_preds.clear()
            self.train_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(y.cpu().numpy())

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_preds:
            metrics = calculate_metrics(
                np.array(self.val_targets), np.array(self.val_preds)
            )
            for name, value in metrics.items():
                self.log(f"val_{name}", value, prog_bar=True)

            self.val_preds.clear()
            self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())

        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self):
        if self.test_preds:
            metrics = calculate_metrics(
                np.array(self.test_targets), np.array(self.test_preds)
            )
            for name, value in metrics.items():
                self.log(f"test_{name}", value)

            logger.info(f"Test metrics: {metrics}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.model.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=5,
            factor=0.5,
            # verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @classmethod
    def safe_load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """Безопасная загрузка модели с обработкой omegaconf"""
        import torch
        import torch.serialization

        # Добавляем безопасные глобалы для omegaconf
        try:
            from omegaconf.base import ContainerMetadata
            from omegaconf.dictconfig import DictConfig

            torch.serialization.add_safe_globals([DictConfig, ContainerMetadata])
        except ImportError:
            pass

        # Пробуем стандартную загрузку
        try:
            return cls.load_from_checkpoint(checkpoint_path, **kwargs)
        except Exception:
            # Если не сработало, пробуем с weights_only=False
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            # Восстанавливаем модель
            if "hyper_parameters" in checkpoint:
                config = checkpoint["hyper_parameters"].get("config")
                if config is None:
                    raise ValueError("Конфиг не найден в чекпойнте")
            else:
                raise ValueError("Некорректный формат чекпойнта")

            # Создаем модель и загружаем веса
            model = cls(config)
            model.load_state_dict(checkpoint["state_dict"])
            return model
