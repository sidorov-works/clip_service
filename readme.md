# CLIP Classification Service

Сервис для zero-shot классификации изображений на основе модели CLIP. Позволяет классифицировать изображения по произвольным текстовым категориям без необходимости дообучения.

## Оглавление

1. [Архитектура](#1-архитектура)
2. [API спецификация](#2-api-спецификация)
3. [Установка и запуск](#3-установка-и-запуск)
4. [Docker](#4-docker)
5. [Конфигурация](#5-конфигурация)
6. [Клиентский код](#6-клиентский-код)
7. [Мониторинг и отладка](#7-мониторинг-и-отладка)

---

## 1. Архитектура

Сервис построен по стандартной схеме с очередью:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Client    │────▶│  FastAPI     │────▶│   Queue     │────▶│  CLIPWorker  │
│  (image)    │◀────│  (HTTP)      │◀────│ (asyncio)   │◀────│  (model)     │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

- **Единственный воркер** — модель CLIP загружена один раз
- **Асинхронная очередь** — задачи обрабатываются последовательно
- **Batch-обработка** — несколько изображений за один HTTP-запрос

### Что такое zero-shot классификация?

Вы сами описываете категории словами, и CLIP определяет, к какой из них относится изображение:

```json
{
  "categories": ["скриншот экрана", "фото товара", "фото упаковки", "сообщение об ошибке"]
}
```

→ Сервис возвращает наиболее подходящую категорию для каждого изображения.

---

## 2. API спецификация

### 2.1 Классификация

#### POST `/classify` — классификация одного изображения

**Запрос:**
```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt_token>" \
  -d '{
    "image": "iVBORw0KGgo...",
    "categories": ["скриншот", "фото товара", "упаковка"]
  }'
```

**Ответ:**
```json
{
  "success": true,
  "task_id": "uuid",
  "category": "скриншот",
  "confidence": 0.95,
  "all_scores": {
    "скриншот": 0.95,
    "фото товара": 0.03,
    "упаковка": 0.02
  },
  "processing_time_ms": 123.4,
  "error": null
}
```

#### POST `/classify/batch` — батчевая классификация

**Запрос:**
```bash
curl -X POST http://localhost:8001/classify/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt_token>" \
  -d '{
    "images": ["iVBORw0KGgo...", "iVBORw0KGgo..."],
    "categories": ["скриншот", "фото товара", "упаковка"]
  }'
```

**Ответ:**
```json
{
  "success": true,
  "task_id": "uuid",
  "results": [
    {
      "success": true,
      "category": "скриншот",
      "confidence": 0.95,
      "all_scores": {...},
      "processing_time_ms": 123.4
    },
    {
      "success": true,
      "category": "фото товара",
      "confidence": 0.88,
      "all_scores": {...},
      "processing_time_ms": 115.2
    }
  ],
  "processing_time_ms": 245.6,
  "error": null
}
```

### 2.2 Health & Info

#### GET `/health`

```bash
curl http://localhost:8001/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model": "openai/clip-vit-base-patch32",
  "queue_size": 0,
  "tasks_processed": 42
}
```

#### GET `/info`

```bash
curl http://localhost:8001/info
```

**Ответ:**
```json
{
  "service": "CLIP Classification Service",
  "version": "1.0.0",
  "model": "openai/clip-vit-base-patch32",
  "device": "cuda",
  "default_categories": [
    "screenshot of computer or phone screen",
    "photo of a product",
    "photo of packaging or box",
    "error message on screen",
    "photo of receipt or document",
    "photo of a person",
    "other image"
  ],
  "max_images_per_batch": 50,
  "max_image_size_mb": 10,
  "queue_maxsize": 100,
  "endpoints": [
    "/classify",
    "/classify/batch",
    "/health",
    "/info"
  ]
}
```

---

## 3. Установка и запуск

### 3.1 Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3.2 Переменные окружения (.env)

```bash
# Сервер
HOST=0.0.0.0
PORT=8001

# Безопасность
INTERNAL_API_SECRET=your_secret_key_here
REQUIRE_AUTH=true

# Модель
MODEL_NAME=openai/clip-vit-base-patch32
MODELS_ROOT=./models

# Device (cuda / mps / cpu)
DEVICE=cuda

# Очередь
QUEUE_MAXSIZE=100
BATCH_TIMEOUT=30.0

# Ограничения
MAX_IMAGES_PER_BATCH=50
MAX_IMAGE_SIZE_MB=10

# Логирование
LOG_PATH=logs
LOGGING_LEVEL=INFO
DOCKER_ENV=false
```

### 3.3 Локальный запуск

```bash
python main.py
```

Сервис запустится на `http://localhost:8001`

### 3.4 Структура проекта

```
clip_service/
├── shared/
│   ├── __init__.py
│   ├── config.py          # Конфигурация
│   ├── schemas.py         # Pydantic модели
│   └── auth_service.py    # Авторизация
├── workers/
│   ├── __init__.py
│   └── clip_worker.py     # Воркер с моделью CLIP
├── main.py                # FastAPI приложение
├── requirements.txt
├── .env
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 4. Docker

### 4.1 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание директории для моделей
RUN mkdir -p /app/models

# Переменные окружения
ENV PYTHONPATH=/app
ENV MODELS_ROOT=/app/models
ENV DOCKER_ENV=true

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 4.2 docker-compose.yml

```yaml
version: '3.8'

services:
  clip-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - MODEL_NAME=openai/clip-vit-base-patch32
      - DEVICE=cuda
      - MODELS_ROOT=/app/models
      - DOCKER_ENV=true
      - QUEUE_MAXSIZE=100
      - MAX_IMAGES_PER_BATCH=50
      - MAX_IMAGE_SIZE_MB=10
      - REQUIRE_AUTH=true
      - INTERNAL_API_SECRET=${INTERNAL_API_SECRET:-your-secret-key}
      - LOG_PATH=/app/logs
      - LOGGING_LEVEL=INFO
    volumes:
      - ./models:/app/models    # Сохраняем скачанные модели между запусками
      - ./logs:/app/logs
      - ./.env:/app/.env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### 4.3 .dockerignore

```dockerignore
__pycache__/
*.pyc
venv/
.venv/
.vscode/
.idea/
logs/
models/
.git/
.gitignore
.env.local
.DS_Store
README.md
```

### 4.4 Сборка и запуск

```bash
# Сборка образа
docker-compose build

# Запуск сервиса
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down

# Перезапуск после изменения .env или конфигов
docker-compose restart
```

### 4.5 Важно про кэш моделей

Модель скачивается **при первом запуске** в папку `/app/models`. Благодаря volume:

```yaml
volumes:
  - ./models:/app/models
```

Папка `models` сохраняется на хосте, поэтому:
- **Первый запуск** → модель скачается (600 MB, ~2-3 минуты)
- **Последующие запуски** → модель уже на диске, загружается мгновенно

При пересборке контейнера модель **не скачивается заново**, если папка `./models` не удалена.

### 4.6 Когда нужна пересборка

| Изменение | Действие |
|-----------|----------|
| `.env` | `docker-compose restart` |
| `config.py` | `docker-compose restart` |
| `main.py` или `workers/*.py` | `docker-compose restart` |
| `requirements.txt` | `docker-compose build --no-cache` |
| `Dockerfile` | `docker-compose build` |

---

## 5. Конфигурация

### Основные параметры

| Параметр | Тип | Значение по умолчанию | Описание |
|----------|-----|----------------------|----------|
| `HOST` | str | `0.0.0.0` | Адрес сервера |
| `PORT` | int | `8001` | Порт сервера |
| `MODEL_NAME` | str | `openai/clip-vit-base-patch32` | Модель CLIP |
| `MODELS_ROOT` | Path | `./models` | Папка для хранения моделей |
| `DEVICE` | str | auto | Устройство (`cuda`/`mps`/`cpu`) |
| `QUEUE_MAXSIZE` | int | `100` | Максимальный размер очереди |
| `BATCH_TIMEOUT` | float | `30.0` | Таймаут обработки (сек) |
| `MAX_IMAGES_PER_BATCH` | int | `50` | Максимум изображений в батче |
| `MAX_IMAGE_SIZE_MB` | int | `10` | Максимальный размер изображения |
| `REQUIRE_AUTH` | bool | `true` | Требовать авторизацию |
| `INTERNAL_API_SECRET` | str | - | Секретный ключ |

### Категории по умолчанию

```python
DEFAULT_CATEGORIES = [
    "screenshot of computer or phone screen",
    "photo of a product",
    "photo of packaging or box",
    "error message on screen",
    "photo of receipt or document",
    "photo of a person",
    "other image"
]
```

**Примечание:** Категории на английском, так как CLIP модели обучались преимущественно на английских текстах.

---

## 6. Клиентский код

### 6.1 Базовый клиент

```python
import httpx
import asyncio

async def classify_image(image_b64: str, categories: list = None):
    async with httpx.AsyncClient() as client:
        payload = {"image": image_b64}
        if categories:
            payload["categories"] = categories
        
        response = await client.post(
            "http://localhost:8001/classify",
            json=payload,
            timeout=30
        )
        return response.json()

# Использование
result = asyncio.run(classify_image("iVBORw0KGgo..."))
print(result["category"], result["confidence"])
```

### 6.2 Батчевый клиент

```python
async def classify_batch(images_b64: list, categories: list = None):
    async with httpx.AsyncClient() as client:
        payload = {"images": images_b64}
        if categories:
            payload["categories"] = categories
        
        response = await client.post(
            "http://localhost:8001/classify/batch",
            json=payload,
            timeout=120
        )
        return response.json()["results"]

# Использование
results = asyncio.run(classify_batch([b64_1, b64_2]))
for r in results:
    print(r["category"], r["confidence"])
```

### 6.3 С авторизацией

```python
import httpx
from jose import jwt
import time

def generate_token(secret: str) -> str:
    payload = {
        "exp": time.time() + 3600,
        "iat": time.time()
    }
    return jwt.encode(payload, secret, algorithm="HS256")

async def classify_with_auth(image_b64: str, secret: str):
    token = generate_token(secret)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/classify",
            json={"image": image_b64},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        return response.json()
```

---

## 7. Мониторинг и отладка

### Проверка здоровья

```bash
curl http://localhost:8001/health
```

### Проверка информации

```bash
curl http://localhost:8001/info
```

### Просмотр логов

```bash
# Локально
tail -f logs/app.log

# Docker
docker-compose logs -f clip-service
```

### Типичные проблемы

| Проблема | Решение |
|----------|---------|
| 401 Unauthorized | Проверьте `INTERNAL_API_SECRET` и заголовок `Authorization` |
| Модель не загружается | Проверьте интернет для скачивания модели (первый запуск) |
| Out of memory | Уменьшите `MAX_IMAGES_PER_BATCH` |
| Таймаут | Увеличьте `BATCH_TIMEOUT` |
| Неверный base64 | Убедитесь, что изображение в правильном формате |

---

## Лицензия

MIT