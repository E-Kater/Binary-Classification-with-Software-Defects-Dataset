#!/usr/bin/env python3
"""Скрипт для запуска MLflow UI"""

import argparse
import subprocess
import sys


def start_mlflow_ui(backend_store_uri: str, port: int = 5000, host: str = "127.0.0.1"):
    """Запуск MLflow UI"""

    cmd = ["mlflow", "ui"]

    if backend_store_uri:
        cmd.extend(["--backend-store-uri", backend_store_uri])

    cmd.extend(["--host", host, "--port", str(port)])

    print(f"Запуск MLflow UI: {' '.join(cmd)}")
    print(f"Доступно по адресу: http://{host}:{port}")  # noqa: E231

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nMLflow UI остановлен")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске MLflow UI: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Запуск MLflow UI")
    parser.add_argument(
        "--port", type=int, default=5000, help="Порт для MLflow UI (по умолчанию: 5000)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Хост для MLflow UI (по умолчанию: 127.0.0.1)",  # noqa501
    )
    parser.add_argument(
        "--backend",
        default="file: ./mlruns",
        help="Backend store URI (по умолчанию: file: ./mlruns)",
    )

    args = parser.parse_args()
    start_mlflow_ui(args.backend, args.port, args.host)


if __name__ == "__main__":
    main()
