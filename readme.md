# CLIP Classification Service

Сервис для zero-shot классификации изображений на основе модели CLIP. Позволяет классифицировать изображения по произвольным текстовым категориям без необходимости дообучения.

## Оглавление

1. [Архитектура](#1-архитектура)
2. [API спецификация](#2-api-спецификация)
3. [Установка и запуск](#3-установка-и-запуск)
4. [Конфигурация](#4-конфигурация)
5. [Клиентский код](#5-клиентский-код)
6. [Интеграция с воркером обработки вложений](#6-интеграция-с-воркером-обработки-вложений)
7. [Мониторинг и отладка](#7-мониторинг-и-отладка)

---

## 1. Архитектура

Сервис построен по аналогии с TEI-сервисом:

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

### 2.1 POST `/classify` — классификация одного изображения

**Запрос:**
```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Secret: your_secret_key" \
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

### 2.2 POST `/classify/batch` — батчевая классификация (основной режим)

**Запрос:**
```bash
curl -X POST http://localhost:8001/classify/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Secret: your_secret_key" \
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

### 2.3 GET `/health` — проверка здоровья

```bash
curl http://localhost:8001/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "device": "mps",
  "model": "openai/clip-vit-base-patch32",
  "queue_size": 0,
  "tasks_processed": 42
}
```

### 2.4 GET `/info` — информация о сервисе

```bash
curl http://localhost:8001/info
```

**Ответ:**
```json
{
  "service": "CLIP Classification Service",
  "version": "1.0.0",
  "model": "openai/clip-vit-base-patch32",
  "device": "mps",
  "default_categories": [
    "скриншот экрана компьютера или телефона",
    "фотография товара",
    "фотография упаковки или коробки",
    "сообщение об ошибке на экране",
    "фотография чека или документа",
    "фотография человека",
    "другое изображение"
  ],
  "max_images_per_batch": 50,
  "max_image_size_mb": 10,
  "queue_maxsize": 100
}
```

---

## 3. Установка и запуск

### 3.1 Установка зависименностей

```bash
pip install torch transformers pillow fastapi uvicorn python-multipart python-dotenv
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

### 3.3 Запуск сервиса

```bash
python main.py
```

Сервис запустится на `http://localhost:8001`

### 3.4 Procfile для Honcho/Overmind

```procfile
classification: uvicorn main:app --host 0.0.0.0 --port ${CLASSIFICATION_PORT:-8001} --workers 1
```

### 3.5 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DEVICE=cuda
ENV DOCKER_ENV=true

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

**requirements.txt:**
```
torch>=2.0.0
transformers>=4.36.0
pillow>=10.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-dotenv>=1.0.0
python-multipart>=0.0.6
```

### 3.6 Структура проекта

```
classification_service/
├── shared/
│   ├── __init__.py
│   ├── config.py          # Конфигурация
│   ├── models.py          # Pydantic модели
│   └── auth_service.py    # Авторизация (общая с другими сервисами)
├── workers/
│   ├── __init__.py
│   └── clip_worker.py     # Воркер с моделью CLIP
├── main.py                # FastAPI приложение
├── requirements.txt
├── .env
└── README.md
```

---

## 4. Конфигурация

### Основные параметры (shared/config.py)

| Параметр | Тип | Значение по умолчанию | Описание |
|----------|-----|----------------------|----------|
| `HOST` | str | `localhost` | Адрес сервера |
| `PORT` | int | `8001` | Порт сервера |
| `MODEL_NAME` | str | `openai/clip-vit-base-patch32` | Модель CLIP |
| `DEVICE` | str | auto (mps/cuda/cpu) | Устройство для инференса |
| `QUEUE_MAXSIZE` | int | `100` | Максимальный размер очереди |
| `BATCH_TIMEOUT` | float | `30.0` | Таймаут обработки батча (сек) |
| `MAX_IMAGES_PER_BATCH` | int | `50` | Максимум изображений в батче |
| `MAX_IMAGE_SIZE_MB` | int | `10` | Максимальный размер изображения (MB) |
| `REQUIRE_AUTH` | bool | `true` | Требовать авторизацию |
| `INTERNAL_API_SECRET` | str | - | Секретный ключ для авторизации |

### Категории по умолчанию

```python
DEFAULT_CATEGORIES = [
    "скриншот экрана компьютера или телефона",
    "фотография товара",
    "фотография упаковки или коробки",
    "сообщение об ошибке на экране",
    "фотография чека или документа",
    "фотография человека",
    "другое изображение"
]
```

---

## 5. Клиентский код

### 5.1 Базовый клиент

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

### 5.2 Батчевый клиент

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

### 5.3 С авторизацией (рекомендуемый)

```python
from http_utils import RetryableHTTPClient, create_signed_client, AuthType

async def classify_with_auth(images_b64: list, secret: str):
    client = create_signed_client(
        RetryableHTTPClient(base_timeout=30, max_retries=2),
        secret=secret,
        auth_type=AuthType.SECRET_HEADER_AUTH
    )
    
    async with client:
        response = await client.post_with_retry(
            "http://localhost:8001/classify/batch",
            json={"images": images_b64}
        )
        return response.json()["results"]
```

---

## 6. Интеграция с воркером обработки вложений

### Добавление классификации в `w_attachments_handler`:

```python
# В __init__ воркера
self.classifier = ClassificationClient(base_url=config.CLASSIFICATION_URL)

# В _process_attachments, перед отправкой в OCR
if pending_attachments:
    images_b64 = [att.data for _, att in pending_attachments]
    
    try:
        classifications = await self.classifier.classify_batch(images_b64)
        
        for (msg, attachment), cls in zip(pending_attachments, classifications):
            if cls["success"] and cls["confidence"] > 0.6:
                logger.info(f"Attachment {attachment.id}: {cls['category']} ({cls['confidence']:.2f})")
                
                # Сохраняем категорию в метаданные (опционально)
                attachment.category = cls["category"]
                attachment.confidence = cls["confidence"]
                
                # Приоритизация на основе категории
                if cls["category"] == "сообщение об ошибке":
                    attachment.priority = "high"
                elif cls["category"] == "скриншот экрана":
                    attachment.priority = "medium"
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        # Продолжаем обработку без классификации
```

### Конфиг для воркера:

```python
# shared/config.py
CLASSIFICATION_URL = os.getenv("CLASSIFICATION_URL", "http://localhost:8001")
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

### Логирование

Сервис логирует:
- Загрузку модели
- Количество обработанных задач
- Ошибки при обработке

### Типичные проблемы

| Проблема | Решение |
|----------|---------|
| 401 Unauthorized | Проверьте `INTERNAL_API_SECRET` и заголовок `X-API-Secret` |
| Модель не загружается | Проверьте интернет для скачивания модели (первый запуск) |
| Out of memory | Уменьшите `MAX_IMAGES_PER_BATCH` |
| Таймаут | Увеличьте `BATCH_TIMEOUT` |
| Неверный base64 | Убедитесь, что изображение в правильном формате |

---

## Преимущества использования

| Аспект | Преимущество |
|--------|-------------|
| **Zero-shot** | Не нужно собирать датасет и дообучать модель |
| **Гибкость** | Категории можно менять на лету в запросе |
| **Производительность** | Batch-обработка до 50 изображений за раз |
| **Масштабирование** | Отдельный сервис не дублирует модель в воркерах |
| **Русский язык** | Категории на русском работают |
| **Авторизация** | Поддержка единой схемы авторизации с другими сервисами |

---

## Лицензия

MIT