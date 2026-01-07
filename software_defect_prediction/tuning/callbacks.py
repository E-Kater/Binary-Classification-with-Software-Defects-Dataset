# src/tuning/callbacks.py
import numpy as np
import optuna
from pytorch_lightning import Callback


class OptunaPruningCallback(Callback):
    """Callback для интеграции Optuna с PyTorch Lightning"""

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_f1"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best_score = -np.inf

    def on_validation_epoch_end(self, trainer, pl_module):
        """Вызывается в конце каждой валидационной эпохи"""
        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            return

        # Сообщаем Optuna о промежуточном результате
        self.trial.report(current_score.item(), step=trainer.current_epoch)

        # Проверяем, нужно ли остановить trial
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {trainer.current_epoch}")

        # Сохраняем лучший результат
        if current_score > self.best_score:
            self.best_score = current_score
