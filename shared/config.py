# shared/config.py
import torch
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:

    # --- Сервер ---
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("CLIP_PORT"))

    # --- Безопасность и аутентификация ---
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")
    ALLOWED_JWT_ALGORITHMS = ["HS256"]
    REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
    
    # --- Модель ---
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
    MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "/app/models")) # ВАЖНО: фиксированная папка, не HF cache

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
    QUEUE_MAXSIZE = 100
    BATCH_TIMEOUT = 30.0  # таймаут обработки батча
    
    # --- Ограничения ---
    MAX_IMAGES_PER_BATCH = 50
    MAX_IMAGE_SIZE_MB = 10
    
    # --- Логирование ---
    LOG_PATH = Path(os.getenv("LOG_PATH", "logs"))
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
    DOCKER_ENV = os.getenv("DOCKER_ENV", "false").lower() == "true"

config = Config()