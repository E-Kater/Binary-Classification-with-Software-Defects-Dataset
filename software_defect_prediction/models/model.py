from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.utils.class_weight import compute_class_weight
from utils.logging import get_logger

from .metric import calculate_metrics

logger = get_logger(__name__)


class DefectClassifier(pl.LightningModule):
    """Модель для классификации дефектов ПО"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=[])

        self.config = config
        model_config = config.model

        # Создание слоев
        layers = []
        input_size = model_config.input_size
        use_batchnorm = getattr(model_config, "use_batchnorm", True)
        for hidden_size in model_config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_config.dropout_rate))
            if not use_batchnorm:
                layers.append(nn.LayerNorm(hidden_size))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, model_config.num_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights()

        self.loss_fn = None
        self.class_weights = None
        # Для метрик
        self.train_preds: List[float] = []
        self.train_targets: List[float] = []
        self.val_preds: List[float] = []
        self.val_targets: List[float] = []
        self.test_preds: List[float] = []
        self.test_targets: List[float] = []

    def _init_weights(self):
        """Инициализация весов для лучшей сходимости"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        smoothing = getattr(self.config.model, "label_smoothing", 0.1)
        if getattr(self.config.model, "use_focal_loss", True):
            from torch.nn import functional as F

            # Focal Loss параметры
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            alpha = (
                class_weights_tensor.to(self.device)
                if class_weights is not None
                else None
            )
            gamma = getattr(self.config.model, "focal_gamma", 2.0)

            def focal_loss(logits, targets):
                ce_loss = F.cross_entropy(logits, targets, reduction="none")
                pt = torch.exp(-ce_loss)
                focal_loss = (
                    (alpha[targets] if alpha is not None else 1.0)
                    * (1 - pt) ** gamma
                    * ce_loss
                )
                return focal_loss.mean()

            self.loss_fn = focal_loss
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device)
                if class_weights is not None
                else None,
                label_smoothing=smoothing,
            )

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.training.optimizer.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Первый период
            T_mult=2,  # Умножение периода
            eta_min=1e-6,  # Минимальный lr
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
            state_dict = checkpoint["state_dict"]
            state_dict.pop("loss_fn.weight")
            model.load_state_dict(checkpoint["state_dict"])
            return model
