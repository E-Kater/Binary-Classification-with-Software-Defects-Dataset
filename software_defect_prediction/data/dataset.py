import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SoftwareDefectDataset(Dataset):
    """Dataset for software defects data"""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor = None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]
