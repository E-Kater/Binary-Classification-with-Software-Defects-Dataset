import logging
import os
import sys
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


try:
    from src.inference.predictor import DefectPredictor
except ImportError:
    sys.path.append(os.path.join(project_root, "software_defect_prediction"))
    from inference.predictor import DefectPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Software Defect Prediction API",
    description="API для предсказания дефектов в программном обеспечении",
    version="1.0.0",
)

# Загрузка модели и предиктора
MODEL_PATH = "models/best_model-v6.ckpt"
SCALER_PATH = "data/processed/scaler.joblib"

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


class PredictionResult(BaseModel):
    """Модель результата предсказания"""

    predictions: list[int]
    is_defects: list[bool]
    defect_probabilities: list[float]
    confidences: list[float]


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске приложения"""
    global predictor

    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Модель не найдена: {MODEL_PATH}")
            return

        # Инициализируем предиктор
        predictor = DefectPredictor(MODEL_PATH)

        if os.path.exists(SCALER_PATH):
            predictor.load_scaler(SCALER_PATH)

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
            "metrics": "/metrics",
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
            "model_info": {"path": MODEL_PATH, "device": str(predictor.device)},
        }

    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
