# CLIP Classification & Validation Service

Сервис для zero-shot классификации изображений и валидации результатов OCR на основе модели CLIP. Позволяет классифицировать изображения по произвольным текстовым категориям и проверять соответствие распознанного текста изображению без необходимости дообучения.

## Оглавление

1. [Архитектура](#1-архитектура)
2. [API спецификация](#2-api-спецификация)
   - 2.1 [Классификация](#21-классификация)
   - 2.2 [Валидация OCR](#22-валидация-ocr)
   - 2.3 [Health & Info](#23-health--info)
3. [Установка и запуск](#3-установка-и-запуск)
4. [Конфигурация](#4-конфигурация)
5. [Клиентский код](#5-клиентский-код)
   - 5.1 [Классификация](#51-классификация)
   - 5.2 [Валидация OCR](#52-валидация-ocr)
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

### Возможности сервиса

| Функция | Описание |
|---------|----------|
| **Классификация** | Определение категории изображения (скриншот, фото товара, чек и т.д.) |
| **Валидация OCR** | Проверка, соответствует ли распознанный текст изображению |

---

## 2. API спецификация

### 2.1 Классификация

#### POST `/classify` — классификация одного изображения

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

#### POST `/classify/batch` — батчевая классификация

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

---

### 2.2 Валидация OCR

#### POST `/validate` — валидация одного результата

**Запрос:**
```bash
curl -X POST http://localhost:8001/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Secret: your_secret_key" \
  -d '{
    "image": "iVBORw0KGgo...",
    "text": "распознанный текст",
    "threshold": 0.5
  }'
```

**Ответ:**
```json
{
  "success": true,
  "task_id": "uuid",
  "is_valid": true,
  "confidence": 0.87,
  "processing_time_ms": 98.7,
  "error": null
}
```

#### POST `/validate/batch` — батчевая валидация

**Запрос:**
```bash
curl -X POST http://localhost:8001/validate/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Secret: your_secret_key" \
  -d '{
    "images": ["iVBORw0KGgo...", "iVBORw0KGgo..."],
    "texts": ["текст1", "текст2"],
    "threshold": 0.5
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
      "is_valid": true,
      "confidence": 0.87,
      "processing_time_ms": 98.7
    },
    {
      "success": true,
      "is_valid": false,
      "confidence": 0.32,
      "processing_time_ms": 95.2
    }
  ],
  "processing_time_ms": 195.0,
  "error": null
}
```

---

### 2.3 Health & Info

#### GET `/health`

```bash
curl http://localhost:8001/health
```

```json
{
  "status": "healthy",
  "device": "mps",
  "model": "openai/clip-vit-base-patch32",
  "queue_size": 0,
  "tasks_processed": 42
}
```

#### GET `/info`

```bash
curl http://localhost:8001/info
```

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
  "queue_maxsize": 100,
  "endpoints": [
    "/classify",
    "/classify/batch",
    "/validate",
    "/validate/batch",
    "/health",
    "/info"
  ]
}
```

---

## 3. Установка и запуск

### 3.1 Установка зависимостей

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
MODELS_ROOT=./models

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
clip-service: uvicorn main:app --host 0.0.0.0 --port ${CLIP_PORT:-8001} --workers 1
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
huggingface-hub>=0.20.0
```

### 3.6 Структура проекта

```
clip_service/
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
| `MODELS_ROOT` | Path | `./models` | Папка для хранения моделей |
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

### Пороги валидации

| Confidence | Оценка | Рекомендация |
|------------|--------|--------------|
| > 0.7 | Отлично | Результат достоверен |
| 0.4 - 0.7 | Средне | Можно использовать с осторожностью |
| < 0.4 | Плохо | Вероятна ошибка OCR, требуется повтор |

---

## 5. Клиентский код

### 5.1 Классификация

```python
# classification_client.py
from http_utils import RetryableHTTPClient, create_signed_client, AuthType
from typing import List


class ClassificationClient:
    """Клиент для классификации изображений."""
    
    def __init__(self, base_url: str = "http://localhost:8001", secret: str = None):
        self.base_url = base_url
        self.secret = secret
    
    def _get_client(self):
        if self.secret:
            return create_signed_client(
                RetryableHTTPClient(base_timeout=30, max_retries=2),
                secret=self.secret,
                auth_type=AuthType.SECRET_HEADER_AUTH
            )
        return RetryableHTTPClient(base_timeout=30, max_retries=2)
    
    async def classify_batch(self, images_b64: List[str], categories: List[str] = None) -> List[dict]:
        """Классифицирует батч изображений."""
        async with self._get_client() as client:
            payload = {"images": images_b64}
            if categories:
                payload["categories"] = categories
            
            response = await client.post_with_retry(
                f"{self.base_url}/classify/batch",
                json=payload
            )
            
            result = response.json()
            if not result.get("success"):
                raise Exception(result.get("error", "Classification failed"))
            
            return result["results"]
```

### 5.2 Валидация OCR

```python
# ocr_validation_client.py
from http_utils import RetryableHTTPClient, create_signed_client, AuthType
from typing import List


class OCRValidationClient:
    """Клиент для валидации OCR результатов."""
    
    def __init__(self, base_url: str = "http://localhost:8001", secret: str = None):
        self.base_url = base_url
        self.secret = self.secret
    
    def _get_client(self):
        if self.secret:
            return create_signed_client(
                RetryableHTTPClient(base_timeout=30, max_retries=2),
                secret=self.secret,
                auth_type=AuthType.SECRET_HEADER_AUTH
            )
        return RetryableHTTPClient(base_timeout=30, max_retries=2)
    
    async def validate_batch(
        self, 
        images_b64: List[str], 
        texts: List[str], 
        threshold: float = 0.5
    ) -> List[dict]:
        """
        Батчевая валидация OCR результатов.
        
        Returns:
            List[dict]: [{"is_valid": bool, "confidence": float}, ...]
        """
        async with self._get_client() as client:
            response = await client.post_with_retry(
                f"{self.base_url}/validate/batch",
                json={
                    "images": images_b64,
                    "texts": texts,
                    "threshold": threshold
                }
            )
            
            result = response.json()
            if not result.get("success"):
                raise Exception(result.get("error", "Validation failed"))
            
            return [
                {"is_valid": r["is_valid"], "confidence": r["confidence"]}
                for r in result["results"]
            ]
```

---

## 6. Интеграция с воркером обработки вложений

### Добавление классификации и валидации в `w_attachments_handler`:

```python
# В __init__ воркера
self.classifier = ClassificationClient(base_url=config.CLASSIFICATION_URL)
self.ocr_validator = OCRValidationClient(base_url=config.CLASSIFICATION_URL)

# В _process_attachments

# 1. Классификация изображений
if pending_attachments:
    images_b64 = [att.data for _, att in pending_attachments]
    
    try:
        classifications = await self.classifier.classify_batch(images_b64)
        
        for (msg, att), cls in zip(pending_attachments, classifications):
            if cls["success"] and cls["confidence"] > 0.6:
                att.category = cls["category"]
                att.confidence = cls["confidence"]
                
                # Приоритизация
                if cls["category"] == "сообщение об ошибке":
                    att.priority = "high"
    except Exception as e:
        logger.error(f"Classification failed: {e}")

# 2. После OCR — валидация
if ocr_results:
    images_b64 = [att.data for _, att in pending_attachments]
    texts = [r["text"] for r in ocr_results if r.get("success")]
    
    try:
        validations = await self.ocr_validator.validate_batch(images_b64, texts, threshold=0.5)
        
        for (msg, att), validation in zip(pending_attachments, validations):
            if validation["is_valid"]:
                logger.info(f"OCR result validated for {att.id}: {validation['confidence']:.2f}")
            else:
                logger.warning(f"OCR validation failed for {att.id}: {validation['confidence']:.2f}")
                # Отправляем на эскалацию или повтор
    except Exception as e:
        logger.error(f"Validation failed: {e}")
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
| **Zero-shot классификация** | Не нужно собирать датасет и дообучать модель |
| **Zero-shot валидация** | CLIP сам определяет, соответствует ли текст изображению |
| **Гибкость** | Категории и пороги можно менять на лету |
| **Производительность** | Batch-обработка до 50 изображений за раз |
| **Масштабирование** | Отдельный сервис не дублирует модель в воркерах |
| **Русский язык** | Категории на русском работают |

---

## Лицензия

MIT