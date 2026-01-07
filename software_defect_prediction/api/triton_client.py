#!/usr/bin/env python3
"""
Клиент для Triton Inference Server
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonClient:
    """Клиент для работы с Triton Inference Server"""

    def __init__(
        self,
        url: str = "localhost:8000",
        protocol: str = "http",  # "http" или "grpc"
        model_name: str = "software_defects_ensemble",
        model_version: str = "1",
    ):
        self.url = url
        self.protocol = protocol
        self.model_name = model_name
        self.model_version = model_version

        # Инициализация клиента
        if protocol == "http":
            self.client = httpclient.InferenceServerClient(url=url)
        elif protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(url=url)
        else:
            raise ValueError(f"Неподдерживаемый протокол: {protocol}")

        logger.info(f"Triton client initialized: {url}, protocol: {protocol}")

    def is_server_ready(self, timeout: int = 30) -> bool:
        """Проверка готовности сервера"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    if self.client.is_server_ready():
                        logger.info("Triton server is ready")
                        return True
                except Exception:
                    pass
                time.sleep(1)

            logger.error(f"Server not ready after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Error checking server readiness: {e}")
            return False

    def is_model_ready(
        self, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> bool:
        """Проверка готовности модели"""
        try:
            model_name = model_name or self.model_name
            model_version = model_version or self.model_version

            if self.client.is_model_ready(model_name, model_version):
                logger.info(f"Model {model_name} v{model_version} is ready")
                return True
            else:
                logger.warning(f"Model {model_name} v{model_version} is not ready")
                return False
        except Exception as e:
            logger.error(f"Error checking model readiness: {e}")
            return False

    def get_model_metadata(self) -> Dict:
        """Получение метаданных модели"""
        try:
            metadata = self.client.get_model_metadata(
                model_name=self.model_name, model_version=self.model_version
            )

            # Конвертируем в словарь
            if self.protocol == "http":
                metadata_dict = {
                    "name": metadata["name"],
                    "versions": metadata["versions"],
                    "platform": metadata["platform"],
                    "inputs": metadata["inputs"],
                    "outputs": metadata["outputs"],
                }
            else:  # gRPC
                metadata_dict = {
                    "name": metadata.name,
                    "versions": list(metadata.versions),
                    "platform": metadata.platform,
                    "inputs": [
                        {
                            "name": inp.name,
                            "datatype": inp.datatype,
                            "shape": list(inp.shape),
                        }
                        for inp in metadata.inputs
                    ],
                    "outputs": [
                        {
                            "name": out.name,
                            "datatype": out.datatype,
                            "shape": list(out.shape),
                        }
                        for out in metadata.outputs
                    ],
                }

            logger.info(f"Model metadata: {metadata_dict}")
            return metadata_dict

        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return {}

    def predict(
        self, data: np.ndarray, raw_features: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Предсказание через Triton

        Args:
            data: Массив с данными (batch_size, features)
            raw_features: Если True, использует сырые фичи (ensemble)

        Returns:
            Словарь с предсказаниями
        """
        try:
            # Подготавливаем входные данные
            if raw_features:
                # Для ensemble модели используем RAW_FEATURES
                input_name = "RAW_FEATURES"
                expected_shape = (data.shape[0], 21)
            else:
                # Для обычной модели
                input_name = "input__0"
                expected_shape = (data.shape[0], 21)

            if data.shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")

            # Создаем входной тензор
            if self.protocol == "http":
                inputs = [httpclient.InferInput(input_name, data.shape, "FP32")]
                inputs[0].set_data_from_numpy(data.astype(np.float32))
            else:  # gRPC
                inputs = [grpcclient.InferInput(input_name, data.shape, "FP32")]
                inputs[0].set_data_from_numpy(data.astype(np.float32))

            # Имена выходов
            if raw_features:
                # Ensemble возвращает PREDICTIONS и PROBABILITIES
                outputs = [
                    httpclient.InferRequestedOutput("PREDICTIONS")
                    if self.protocol == "http"
                    else grpcclient.InferRequestedOutput("PREDICTIONS"),
                    httpclient.InferRequestedOutput("PROBABILITIES")
                    if self.protocol == "http"
                    else grpcclient.InferRequestedOutput("PROBABILITIES"),
                ]
            else:
                # Обычная модель возвращает output__0
                outputs = [
                    httpclient.InferRequestedOutput("output__0")
                    if self.protocol == "http"
                    else grpcclient.InferRequestedOutput("output__0")
                ]

            # Выполняем inference
            start_time = time.time()

            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
            )

            inference_time = time.time() - start_time

            # Получаем результаты
            result = {}

            if raw_features:
                predictions = response.as_numpy("PREDICTIONS")
                probabilities = response.as_numpy("PROBABILITIES")

                result = {
                    "predictions": np.argmax(predictions, axis=1),
                    "probabilities": probabilities,
                    "defect_probability": probabilities[:, 1],
                    "inference_time": inference_time,
                }
            else:
                output = response.as_numpy("output__0")

                result = {
                    "logits": output,
                    "probabilities": self._softmax(output),
                    "predictions": np.argmax(output, axis=1),
                    "defect_probability": self._softmax(output)[:, 1],
                    "inference_time": inference_time,
                }

            logger.info(f"Inference completed in {inference_time: .4f}s")
            return result

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def predict_single(self, features: Dict[str, float]) -> Dict:
        """Предсказание для одного образца"""
        # Конвертируем в массив
        feature_names = [
            "loc",
            "v(g)",
            "ev(g)",
            "iv(g)",
            "n",
            "v",
            "l",
            "d",
            "i",
            "e",
            "b",
            "t",
            "lOCode",
            "lOComment",
            "lOBlank",
            "locCodeAndComment",
            "uniq_Op",
            "uniq_Opnd",
            "total_Op",
            "total_Opnd",
            "branchCount",
        ]

        # Создаем массив в правильном порядке
        data_array = np.array(
            [[features.get(name, 0.0) for name in feature_names]], dtype=np.float32
        )

        # Предсказание
        result = self.predict(data_array, raw_features=True)

        return {
            "prediction": int(result["predictions"][0]),
            "is_defect": bool(result["predictions"][0] == 1),
            "defect_probability": float(result["defect_probability"][0]),
            "confidence": float(np.max(result["probabilities"][0])),
            "inference_time": result["inference_time"],
        }

    def batch_predict(self, data_list: List[Dict[str, float]]) -> List[Dict]:
        """Пакетное предсказание"""
        feature_names = [
            "loc",
            "v(g)",
            "ev(g)",
            "iv(g)",
            "n",
            "v",
            "l",
            "d",
            "i",
            "e",
            "b",
            "t",
            "lOCode",
            "lOComment",
            "lOBlank",
            "locCodeAndComment",
            "uniq_Op",
            "uniq_Opnd",
            "total_Op",
            "total_Opnd",
            "branchCount",
        ]

        # Создаем batch
        batch_data = np.array(
            [
                [features.get(name, 0.0) for name in feature_names]
                for features in data_list
            ],
            dtype=np.float32,
        )

        # Предсказание
        result = self.predict(batch_data, raw_features=True)

        # Формируем результаты
        results = []
        for i in range(len(data_list)):
            results.append(
                {
                    "prediction": int(result["predictions"][i]),
                    "is_defect": bool(result["predictions"][i] == 1),
                    "defect_probability": float(result["defect_probability"][i]),
                    "confidence": float(np.max(result["probabilities"][i])),
                    "sample_id": i,
                }
            )

        return results

    def benchmark(self, num_requests: int = 100, batch_size: int = 1):
        """Бенчмарк производительности"""
        logger.info(
            f"Running benchmark: {num_requests} requests, batch_size={batch_size}"
        )

        # Создаем тестовые данные
        test_data = np.random.randn(num_requests * batch_size, 21).astype(np.float32)

        latencies = []

        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size]

            start_time = time.time()
            result = self.predict(batch, raw_features=True)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

            if (i // batch_size) % 10 == 0:
                logger.info(
                    f"Processed {i + batch_size}/{len(test_data)} samples.Result {result}"
                )

        # Статистика
        latencies = np.array(latencies)

        stats = {
            "total_requests": len(latencies),
            "total_samples": len(test_data),
            "batch_size": batch_size,
            "avg_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "throughput": len(test_data) / np.sum(latencies),
        }

        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        for key, value in stats.items():
            if "latency" in key:
                logger.info(f"{key}: {value * 1000: .2f} ms")
            elif "throughput" in key:
                logger.info(f"{key}: {value: .2f} samples/sec")
            else:
                logger.info(f"{key}: {value}")

        return stats

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax функция"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def main():
    """Пример использования клиента"""
    print("=" * 60)
    print("Triton Inference Server Client")
    print("=" * 60)

    # Инициализация клиента
    client = TritonClient(
        url="localhost:8000", protocol="http", model_name="software_defects_ensemble"
    )

    # Проверка сервера
    if not client.is_server_ready():
        print("❌ Сервер не готов")
        return

    # Проверка модели
    if not client.is_model_ready():
        print("❌ Модель не готова")
        return

    # Получение метаданных
    metadata = client.get_model_metadata()
    print(f"\nМодель: {metadata.get('name', 'N/A')}")
    print(f"Платформа: {metadata.get('platform', 'N/A')}")

    # Пример предсказания
    print("\nПример предсказания:")

    sample_features = {
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

    try:
        result = client.predict_single(sample_features)

        status = "ДЕФЕКТ" if result["is_defect"] else "БЕЗ ДЕФЕКТА"
        print(f"  Результат: {status}")
        print(f"  Вероятность дефекта: {result['defect_probability']: .3f}")
        print(f"  Уверенность: {result['confidence']: .3f}")
        print(f"  Время inference: {result['inference_time'] * 1000: .2f} ms")

    except Exception as e:
        print(f"  Ошибка: {e}")

    # Бенчмарк (опционально)
    print("\n" + "=" * 60)
    run_benchmark = input("Запустить бенчмарк производительности? (y/n): ")

    if run_benchmark.lower() == "y":
        print("\nЗапуск бенчмарка...")
        stats = client.benchmark(num_requests=50, batch_size=4)

        # Сохраняем результаты
        import json

        with open("triton_benchmark_results.json", "w") as f:
            json.dump(stats, f, indent=2)

        print("\nРезультаты сохранены в triton_benchmark_results.json")

    print("\n" + "=" * 60)
    print("Клиент Triton готов к работе!")
    print("=" * 60)


if __name__ == "__main__":
    main()
