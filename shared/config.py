# shared/config.py
"""
Конфигурация CLIP Classification Service.
Поддерживает авторизацию, фиксированную папку для моделей, настройки очереди.
"""

import torch
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    # --- Сервер ---
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", "8001"))

    # --- Безопасность и аутентификация ---
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")
    ALLOWED_JWT_ALGORITHMS = ["HS256"]
    REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"

    # --- Модель ---
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")

    # --- Пути для моделей (фиксированная папка, не HF cache) ---
    MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "./models"))
    # Формируем путь для CLIP модели
    MODEL_PATH = MODELS_ROOT / MODEL_NAME.replace('/', '--')

    # --- Device (mps / cuda / cpu) ---
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    # --- Категории по умолчанию (можно переопределить в запросе) ---
    DEFAULT_CATEGORIES = [
        "скриншот экрана компьютера или телефона",
        "фотография товара",
        "фотография упаковки или коробки",
        "сообщение об ошибке на экране",
        "фотография чека или документа",
        "фотография человека",
        "другое изображение"
    ]

    # --- Очередь ---
    QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE", "100"))
    BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "30.0"))  # таймаут обработки батча

    # --- Ограничения ---
    MAX_IMAGES_PER_BATCH = int(os.getenv("MAX_IMAGES_PER_BATCH", "50"))
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))

    # --- Логирование ---
    LOG_PATH = Path(os.getenv("LOG_PATH", "logs"))
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
    DOCKER_ENV = os.getenv("DOCKER_ENV", "false").lower() == "true"


config = Config()