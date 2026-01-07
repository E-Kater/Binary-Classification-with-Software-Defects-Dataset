import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.append("software_defect_prediction")

from models.model import DefectClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def convert_to_onnx(cfg: DictConfig):
    """
    Конвертация PyTorch модели в ONNX

    Args:
        include_preprocess: Если True, добавляет препроцессинг в граф
        :param cfg:
    """
    checkpoint_path = cfg.model.model_path
    include_preprocess = cfg.model.include_preprocess
    model = DefectClassifier.safe_load_from_checkpoint(checkpoint_path)
    onnx_path = cfg.model.onnx_model_path
    model.eval()

    # Пример входных данных
    if include_preprocess:
        # Для модели с препроцессингом
        dummy_input = torch.randn(1, 3, 224, 224)  # RGB изображение
    else:
        # Для чистого инференса
        dummy_input = torch.randn(1, cfg.model.input_size)

    # Экспорт в ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=cfg.model.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    print(f"Модель успешно конвертирована: {onnx_path}")
    print(f"Размер: {Path(onnx_path).stat().st_size / 1024 / 1024: .2f} MB")

    return onnx_path


if __name__ == "__main__":
    convert_to_onnx()
