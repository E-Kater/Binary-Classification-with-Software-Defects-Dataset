import os
import sys
from typing import Optional

import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging import get_logger  # noqa402

from .dataset import SoftwareDefectDataset  # noqa402

logger = get_logger(__name__)


class DefectDataModule(pl.LightningDataModule):
    """DataModule для загрузки и обработки данных"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config.data
        self.preprocess_config = config.preprocessing

        self.scaler = StandardScaler()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Загрузка и сохранение данных"""
        if not os.path.exists(self.data_config.raw_path):
            logger.error(f"Файл данных не найден: {self.data_config.raw_path}")
            raise FileNotFoundError(
                f"Файл данных не найден: {self.data_config.raw_path}"
            )  # noqa501

    def setup(self, stage: Optional[str] = None):
        """Настройка датасетов для разных стадий"""
        if stage in ("fit", None):
            self.prepare_train_val_data()
        if stage in ("test", None):
            self._prepare_test_data()

    def prepare_train_val_data(self):
        """Подготовка train/val данных"""
        # Загрузка данных
        df = pd.read_csv(self.data_config.raw_path)
        # Разделение на фичи и таргет
        X = df[self.preprocess_config.numeric_features].copy()
        y = df[self.data_config.target_col].map({True: 1, False: 0})
        # Разделение на train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.data_config.test_size,
            random_state=self.data_config.random_state,
            stratify=y,
        )

        val_size_adjusted = self.data_config.val_size / (1 - self.data_config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.data_config.random_state,
            stratify=y_temp,
        )

        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Сохранение тестовых данных
        X_test_scaled = self.scaler.transform(X_test)
        test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_data[self.data_config.target_col] = y_test.values
        test_data.to_csv(f"{self.data_config.processed_path}test.csv", index=False)

        # Сохранение scaler
        joblib.dump(self.scaler, f"{self.data_config.processed_path}scaler.joblib")

        # Создание датасетов
        self.train_dataset = SoftwareDefectDataset(
            torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train.values)
        )
        self.val_dataset = SoftwareDefectDataset(
            torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val.values)
        )

        logger.info(f"Train size: {len(self.train_dataset)}")
        logger.info(f"Val size: {len(self.val_dataset)}")

    def _prepare_test_data(self):
        """Подготовка тестовых данных"""
        test_path = f"{self.data_config.processed_path}test.csv"
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            X_test = test_df[self.preprocess_config.numeric_features]
            y_test = test_df[self.data_config.target_col]

            self.test_dataset = SoftwareDefectDataset(
                torch.FloatTensor(X_test.values), torch.LongTensor(y_test.values)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.config.model.batch_size,
                shuffle=False,
                num_workers=4,
            )
