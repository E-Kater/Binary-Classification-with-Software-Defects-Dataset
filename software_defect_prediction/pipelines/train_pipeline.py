import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from training.trainer import ModelTrainer


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Основной пайплайн обучения"""

    # Создание директорий
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Обучение модели
    trainer = ModelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
