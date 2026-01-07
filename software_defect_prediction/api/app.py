import logging
import os
import sys
from typing import List

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "software_defect_prediction"))

from inference.predictor import DefectPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Software Defect Prediction API",
    description="API для предсказания дефектов в программном обеспечении",
    version="1.0.0",
)

config_path: str = "configs/model_config.yaml"
config = OmegaConf.load(config_path)
model_path = config.model.model_path
session = ort.InferenceSession(str(config.model.onnx_model_path))
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
scaler_path = config.model.scaler_path

predictor = None


class FeatureInput(BaseModel):
    """Модель для ввода одного образца"""

    loc: float
    v_g: float = Field(alias="v(g)")
    ev_g: float = Field(alias="ev(g)")
    iv_g: float = Field(alias="iv(g)")
    n: float
    v: float
    l: float
    d: float
    i: float
    e: float
    b: float
    t: float
    lOCode: int
    lOComment: int
    lOBlank: int
    locCodeAndComment: int
    uniq_Op: float
    uniq_Opnd: float
    total_Op: float
    total_Opnd: float
    branchCount: float

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "loc": 22.0,
                "v(g)": 3.0,
                "ev(g)": 1.0,
                "iv(g)": 2.0,
                "n": 60.0,
                "v": 278.63,
                "l": 0.06,
                "d": 19.56,
                "i": 14.25,
                "e": 5448.79,
                "b": 0.09,
                "t": 302.71,
                "lOCode": 17,
                "lOComment": 1,
                "lOBlank": 1,
                "locCodeAndComment": 0,
                "uniq_Op": 16.0,
                "uniq_Opnd": 9.0,
                "total_Op": 38.0,
                "total_Opnd": 22.0,
                "branchCount": 5.0,
            }
        }


class BatchInput(BaseModel):
    """Модель для пакетного ввода"""

    samples: List[FeatureInput]


class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""

    model_name: str = "defect_classifier"
    model_version: str = "1"
    outputs: List[List[float]]


class PredictionResult(BaseModel):
    """Модель результата предсказания"""

    predictions: list[int]
    is_defects: list[bool]
    defect_probabilities: list[float]
    confidences: list[float]


class PredictionRequest(BaseModel):
    """Запрос на предсказание"""

    inputs: List[List[float]] = [
        [
            22.0,
            3.0,
            1.0,
            2.0,
            60.0,
            278.63,
            0.06,
            19.56,
            14.25,
            5448.79,
            0.09,
            302.71,
            17,
            1,
            1,
            0,
            16.0,
            9.0,
            38.0,
            22.0,
            5.0,
        ]
    ]


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске приложения"""
    global predictor

    try:
        if not os.path.exists(model_path):
            logger.error(f"Модель не найдена: {model_path}")
            return

        # Инициализируем предиктор
        predictor = DefectPredictor(model_path)

        if os.path.exists(scaler_path):
            predictor.load_scaler(scaler_path)

        logger.info("Модель успешно загружена")

    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Software Defect Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "predict_onnx": "/onnx/predict",
            "metrics": "/metrics",
            "model_info": "/model_info",
        },
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    status = predictor is not None
    return {
        "status": "healthy" if status else "unhealthy",
        "model_loaded": status,
        "device": str(predictor.device) if predictor else None,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_single(features: FeatureInput):
    """Предсказание для одного образца"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        # Конвертируем Pydantic модель в словарь
        features_dict = features.model_dump(by_alias=True)

        # Предсказание
        result = predictor.predict(pd.DataFrame([features_dict]))

        return PredictionResult(**result)

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=PredictionResult)
async def predict_batch(batch_input: BatchInput):
    """Пакетное предсказание"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        # Конвертируем в список словарей
        samples = [sample.model_dump(by_alias=True) for sample in batch_input.samples]

        # Создаем DataFrame
        df = pd.DataFrame(samples)

        # Предсказание
        predictions = predictor.predict(df)

        return PredictionResult(**predictions)

    except Exception as e:
        logger.error(f"Ошибка пакетного предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/onnx/predict", response_model=PredictionResponse)
async def predict_onnx(request: PredictionRequest):
    try:
        # Преобразуем входные данные
        input_data = np.array(request.inputs, dtype=np.float32)

        # Выполняем предсказание
        outputs = session.run([output_name], {input_name: input_data})[0]

        return PredictionResponse(outputs=outputs.tolist())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Информация о загруженной модели"""
    return {
        "model_path": str(model_path),
        "input_name": config.model.name,
        "input_size": config.model.input_size,
        "dropout_rate": config.model.dropout_rate,
        "learning_rate": config.model.learning_rate,
        "batch_size": config.model.batch_size,
    }


@app.get("/metrics")
async def get_metrics():
    """Получение метрик модели на тестовых данных"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        test_data_path = "data/processed/test.csv"
        if not os.path.exists(test_data_path):
            raise HTTPException(status_code=404, detail="Тестовые данные не найдены")

        metrics = predictor.evaluate_on_test(test_data_path)

        return {
            "test_metrics": metrics,
            "model_info": {"path": model_path, "device": str(predictor.device)},
        }

    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
