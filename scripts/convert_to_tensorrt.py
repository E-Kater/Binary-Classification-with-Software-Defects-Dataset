#!/usr/bin/env python
"""Скрипт для конвертации модели в TensorRT формат"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.append("software_defect_prediction")

from models.model import DefectClassifier

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def convert_to_tensorrt(cfg: DictConfig):
    """
    Конвертация PyTorch модели в TensorRT

    Args:
        cfg: Конфигурация из Hydra
    """
    try:
        # Проверяем наличие TensorRT
        import tensorrt as trt

        logger.info(f"TensorRT версия: {trt.__version__}")
    except ImportError:
        logger.error("TensorRT не установлен. Установите: pip install tensorrt")
        return None

    # Получаем параметры из конфига
    checkpoint_path = cfg.model.model_path
    input_size = cfg.model.input_size
    batch_size = cfg.tensorrt.get("batch_size", 1)
    precision = cfg.tensorrt.get("precision", "fp32")
    workspace_size = cfg.tensorrt.get("workspace_size", 1024)  # MB
    output_dir = Path(cfg.tensorrt.get("output_dir", "models/tensorrt"))

    # Создаем директории
    output_dir.mkdir(parents=True, exist_ok=True)

    # Генерируем имя выходного файла
    model_name = Path(checkpoint_path).stem
    engine_path = output_dir / f"{model_name}_{precision}.engine"
    onnx_temp_path = output_dir / f"{model_name}_temp.onnx"

    logger.info("=" * 60)
    logger.info("TensorRT Conversion")
    logger.info("=" * 60)
    logger.info(f"Model: {checkpoint_path}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Output: {engine_path}")

    # Шаг 1: Загрузка PyTorch модели
    logger.info("1. Загрузка PyTorch модели...")
    try:
        model = DefectClassifier.safe_load_from_checkpoint(checkpoint_path)
        model.eval()
        logger.info("✓ Модель загружена успешно")
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки модели: {e}")
        return None

    # Шаг 2: Экспорт в ONNX (промежуточный формат)
    logger.info("2. Экспорт в ONNX...")
    try:
        dummy_input = torch.randn(batch_size, input_size)

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_temp_path),
            export_params=True,
            opset_version=cfg.model.get("opset_version", 13),
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if cfg.tensorrt.get("dynamic_batch", False)
            else None,
            verbose=cfg.tensorrt.get("verbose", False),
        )

        onnx_size = onnx_temp_path.stat().st_size / 1024 / 1024
        logger.info(f"ONNX модель сохранена: {onnx_temp_path} ({onnx_size: .2f} MB)")

    except Exception as e:
        logger.error(f"✗ Ошибка экспорта в ONNX: {e}")
        return None

    # Шаг 3: Конвертация в TensorRT
    logger.info("3. Конвертация в TensorRT...")
    try:
        engine = build_tensorrt_engine(
            onnx_path=str(onnx_temp_path),
            engine_path=str(engine_path),
            precision=precision,
            batch_size=batch_size,
            workspace_size=workspace_size,
            dynamic_batch=cfg.tensorrt.get("dynamic_batch", False),
            max_batch_size=cfg.tensorrt.get("max_batch_size", 32),
            min_batch_size=cfg.tensorrt.get("min_batch_size", 1),
            opt_batch_size=cfg.tensorrt.get("opt_batch_size", 16),
        )

        if engine:
            engine_size = engine_path.stat().st_size / 1024 / 1024
            logger.info(
                f"✓ TensorRT engine сохранен: {engine_path} ({engine_size: .2f} MB)"
            )

            # Удаляем временный ONNX файл
            if not cfg.tensorrt.get("keep_onnx", False):
                onnx_temp_path.unlink()
                logger.info("✓ Временный ONNX файл удален")

            # Сохраняем информацию о конвертации
            save_conversion_info(cfg, str(engine_path), engine)

            return str(engine_path)
        else:
            logger.error("✗ Не удалось создать TensorRT engine")
            return None

    except Exception as e:
        logger.error(f"✗ Ошибка конвертации в TensorRT: {e}")
        # Оставляем ONNX файл для отладки
        logger.info(f"ONNX файл сохранен для отладки: {onnx_temp_path}")
        return None


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp32",
    batch_size: int = 1,
    workspace_size: int = 1024,
    dynamic_batch: bool = False,
    max_batch_size: int = 32,
    min_batch_size: int = 1,
    opt_batch_size: int = 16,
):
    """
    Строит TensorRT engine из ONNX модели

    Args:
        onnx_path: Путь к ONNX модели
        engine_path: Путь для сохранения engine
        precision: Точность (fp32, fp16, int8)
        batch_size: Размер батча
        workspace_size: Размер workspace в MB
        dynamic_batch: Использовать динамический батч
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Читаем ONNX модель
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parser error: {parser.get_error(error)}")
            return None

    logger.info(f"✓ ONNX модель прочитана, слоев: {network.num_layers}")

    # Настройка конфигурации
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * 1024 * 1024  # Convert to bytes

    # Настройка precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("✓ Используется FP16 precision")
        else:
            logger.warning("Платформа не поддерживает FP16, используется FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Для INT8 нужен calibrator
            logger.info("✓ Используется INT8 precision (требуется calibrator)")
        else:
            logger.warning("Платформа не поддерживает INT8, используется FP32")

    # Настройка профилей для динамического батча
    if dynamic_batch:
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape

        # Устанавливаем диапазоны размеров батча
        profile.set_shape(
            input_tensor.name,
            (min_batch_size, *input_shape[1:]),  # min
            (opt_batch_size, *input_shape[1:]),  # opt
            (max_batch_size, *input_shape[1:]),  # max
        )
        config.add_optimization_profile(profile)
        logger.info(
            f"✓ Динамический батч: {min_batch_size}-{opt_batch_size}-{max_batch_size}"
        )
    else:
        builder.max_batch_size = batch_size

    # Строим engine
    logger.info("Построение TensorRT engine...")
    engine = builder.build_engine(network, config)

    if engine:
        # Сохраняем engine
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        # Получаем информацию о engine
        logger.info(" TensorRT engine создан")
        logger.info(f"  Входные данные: {engine.get_binding_shape(0)}")
        logger.info(f"  Выходные данные: {engine.get_binding_shape(1)}")
        logger.info(f"  Количество слоев: {engine.num_layers}")

        return engine
    else:
        logger.error("Не удалось построить TensorRT engine")
        return None


def save_conversion_info(cfg: DictConfig, engine_path: str, engine):
    """Сохраняет информацию о конвертации"""
    import json
    from datetime import datetime

    info = {
        "conversion_date": datetime.now().isoformat(),
        "original_model": cfg.model.model_path,
        "tensorrt_engine": str(engine_path),
        "input_size": cfg.model.input_size,
        "batch_size": cfg.tensorrt.get("batch_size", 1),
        "precision": cfg.tensorrt.get("precision", "fp32"),
        "dynamic_batch": cfg.tensorrt.get("dynamic_batch", False),
        "workspace_size_mb": cfg.tensorrt.get("workspace_size", 1024),
        "engine_info": {
            "num_layers": engine.num_layers,
            "max_batch_size": engine.max_batch_size,
            "device_memory_size_mb": engine.device_memory_size / 1024 / 1024,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    info_path = Path(engine_path).with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)

    logger.info(f"✓ Информация о конвертации сохранена: {info_path}")

    return info_path


def validate_tensorrt_engine(engine_path: str, input_size: int, batch_size: int = 1):
    """
    Валидация TensorRT engine

    Args:
        engine_path: Путь к TensorRT engine
        input_size: Размер входных данных
        batch_size: Размер батча
    """
    try:
        import pycuda.driver as cuda
        import tensorrt as trt

        # Загружаем engine
        logger.info("Валидация TensorRT engine...")
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        # Создаем execution context
        context = engine.create_execution_context()

        # Подготавливаем входные данные
        input_shape = (batch_size, input_size)
        output_shape = (batch_size, 2)  # Предполагаем бинарную классификацию

        # Выделяем память на GPU
        d_input = cuda.mem_alloc(batch_size * input_size * 4)  # float32
        d_output = cuda.mem_alloc(batch_size * 2 * 4)  # float32

        # Создаем stream
        stream = cuda.Stream()

        # Генерируем тестовые данные
        h_input = np.random.randn(*input_shape).astype(np.float32)
        h_output = np.empty(output_shape, dtype=np.float32)

        # Копируем данные на GPU
        cuda.memcpy_htod_async(d_input, h_input, stream)

        # Запускаем inference
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
        )

        # Копируем результаты обратно
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        logger.info("✓ TensorRT engine успешно прошел валидацию")
        logger.info(f"  Вход: {h_input.shape}, Выход: {h_output.shape}")
        logger.info(f"  Время inference: {context.execution_time} ms")

        # Освобождаем память
        d_input.free()
        d_output.free()

        return True

    except ImportError:
        logger.warning("Pycuda не установлен, пропускаем валидацию на GPU")
        logger.warning("Установите: pip install pycuda")
        return None
    except Exception as e:
        logger.error(f"✗ Ошибка валидации TensorRT engine: {e}")
        return False


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def convert_and_validate(cfg: DictConfig):
    """Конвертация и валидация TensorRT модели"""
    engine_path = convert_to_tensorrt(cfg)

    if engine_path and cfg.tensorrt.get("validate", True):
        validate_tensorrt_engine(
            engine_path=engine_path,
            input_size=cfg.model.input_size,
            batch_size=cfg.tensorrt.get("batch_size", 1),
        )

    return engine_path


if __name__ == "__main__":
    # Запуск конвертации
    engine_path = convert_to_tensorrt()

    if engine_path:
        print("\n" + "=" * 60)
        print("TensorRT конвертация завершена успешно!")
        print(f"Engine сохранен: {engine_path}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TensorRT конвертация завершена с ошибками")
        print("=" * 60)
