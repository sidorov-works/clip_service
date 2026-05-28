# main.py
"""
CLIP Classification Service — сервер для zero-shot классификации изображений и валидации OCR.

Архитектура:
- Единственный воркер с загруженной моделью CLIP
- Асинхронная очередь для batch-запросов
- Поддержка batch-обработки нескольких изображений за один запрос

Эндпоинты:
- POST /classify — классификация одного изображения
- POST /classify/batch — батчевая классификация (основной режим)
- POST /validate — валидация одного OCR результата
- POST /validate/batch — батчевая валидация OCR результатов
- GET /health — проверка здоровья
- GET /info — информация о сервисе
"""

import asyncio
import base64
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from shared.config import config
from shared.models import (
    # Pydantic модели для API
    ClassifyRequest, 
    ClassifyResponse,
    BatchClassifyRequest, 
    BatchClassifyResponse,
    HealthResponse,
    # Внутренние модели для очереди
    ClassifyTask, 
    ClassifyResult,
)
from workers.clip_worker import CLIPWorker
from shared.auth_service import require_header_secret

from pathlib import Path

from logger_utils import get_logger
logger = get_logger(
    name="CLIP SERVICE",
    level="DEBUG",
    log_file=Path("logs") / "app.log",
    docker_mode=False
)

import setproctitle
setproctitle.setproctitle("classification_service")

# Авторизация
require_auth = require_header_secret if config.REQUIRE_AUTH else lambda: None

# ======================================================================
# Глобальные объекты (как в TEI)
# ======================================================================

input_queue: asyncio.Queue = None      # Очередь входящих задач
output_queue: asyncio.Queue = None     # Очередь исходящих результатов
worker: CLIPWorker = None              # Единственный воркер с моделью
dispatcher: Dict[str, asyncio.Future] = {}  # task_id -> Future


# ======================================================================
# Вспомогательные функции
# ======================================================================

def decode_base64_image(b64_str: str) -> bytes:
    """
    Декодирует base64-строку в байты.
    Поддерживает формат с префиксом "data:image/...;base64," и без него.
    """
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    
    try:
        return base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}")


def validate_image_size(image_bytes: bytes) -> None:
    """Проверяет размер изображения, выбрасывает 413 если слишком большой."""
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > config.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            413,
            f"Image too large. Max {config.MAX_IMAGE_SIZE_MB} MB, got {size_mb:.2f} MB"
        )


async def submit_classify_task(image_bytes: bytes, categories: List[str]) -> ClassifyResult:
    """
    Отправляет задачу классификации в очередь и ожидает результат.
    """
    if input_queue.qsize() >= config.QUEUE_MAXSIZE * 0.9:
        raise HTTPException(503, "Service busy, queue is full")
    
    task_id = str(uuid.uuid4())
    future = asyncio.get_event_loop().create_future()
    dispatcher[task_id] = future
    
    task = ClassifyTask(
        task_id=task_id,
        image_bytes=image_bytes,
        categories=categories,
        created_at=time.time()
    )
    
    try:
        await input_queue.put(task)
        result = await asyncio.wait_for(future, timeout=config.BATCH_TIMEOUT)
        return result
    except asyncio.TimeoutError:
        dispatcher.pop(task_id, None)
        if not future.done():
            future.cancel()
        raise HTTPException(504, f"Classification timeout after {config.BATCH_TIMEOUT} seconds")
    except Exception as e:
        dispatcher.pop(task_id, None)
        raise


# ======================================================================
# Lifespan (как в TEI)
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global input_queue, output_queue, worker
    
    # Инициализируем очереди
    input_queue = asyncio.Queue(maxsize=config.QUEUE_MAXSIZE)
    output_queue = asyncio.Queue(maxsize=config.QUEUE_MAXSIZE)
    
    # Создаём и запускаем воркера
    worker = CLIPWorker(input_queue, output_queue)
    worker_task = asyncio.create_task(worker.start())
    
    # Запускаем диспетчер результатов
    dispatcher_task = asyncio.create_task(result_dispatcher())
    
    # Ждём загрузки модели
    while not worker.is_healthy():
        await asyncio.sleep(0.1)
    
    print(f"Classification сервер запущен. Модель: {config.MODEL_NAME}, устройство: {config.DEVICE}")
    
    yield
    
    # Корректная остановка
    worker.running = False
    worker_task.cancel()
    dispatcher_task.cancel()
    await asyncio.gather(worker_task, dispatcher_task, return_exceptions=True)
    print("Classification сервер остановлен")


async def result_dispatcher():
    """Отправляет результаты обратно ожидающим клиентам."""
    while True:
        try:
            result = await output_queue.get()
            future = dispatcher.pop(result.task_id, None)
            if future and not future.done():
                future.set_result(result)
            output_queue.task_done()
        except Exception as e:
            logger.error(f"Ошибка в диспетчере: {e}")


# ======================================================================
# FastAPI приложение
# ======================================================================

app = FastAPI(
    lifespan=lifespan,
    title="CLIP Classification Service",
    description="Zero-shot классификация изображений и валидация OCR через CLIP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# Эндпоинты классификации
# ======================================================================

@app.post("/classify", response_model=ClassifyResponse)
async def classify_single(
    request: Request,
    req: ClassifyRequest,
    _ = Depends(require_auth)
):
    """
    Классификация одного изображения.
    
    Request body:
    {
        "image": "base64...",
        "categories": ["категория1", "категория2"]  # опционально
    }
    """
    start_time = time.time()
    
    categories = req.categories or config.DEFAULT_CATEGORIES
    
    image_bytes = decode_base64_image(req.image)
    validate_image_size(image_bytes)
    
    result = await submit_classify_task(image_bytes, categories)
    
    return ClassifyResponse(
        success=result.success,
        task_id=result.task_id,
        category=result.category,
        confidence=result.confidence,
        all_scores=result.all_scores or {},
        processing_time_ms=result.processing_time_ms,
        error=result.error
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(
    request: Request,
    req: BatchClassifyRequest,
    _ = Depends(require_auth)
):
    """
    Батчевая классификация нескольких изображений (основной режим).
    
    Request body:
    {
        "images": ["base64...", "base64..."],
        "categories": ["категория1", "категория2"]  # опционально
    }
    """
    start_time = time.time()
    
    images = req.images
    if not images:
        raise HTTPException(400, "images list is required and cannot be empty")
    
    if len(images) > config.MAX_IMAGES_PER_BATCH:
        raise HTTPException(
            400,
            f"Too many images in batch. Max {config.MAX_IMAGES_PER_BATCH}, got {len(images)}"
        )
    
    categories = req.categories or config.DEFAULT_CATEGORIES
    task_id = str(uuid.uuid4())
    
    # Подготовка задач
    tasks = []
    for i, image_b64 in enumerate(images):
        try:
            image_bytes = decode_base64_image(image_b64)
            validate_image_size(image_bytes)
            tasks.append(submit_classify_task(image_bytes, categories))
        except HTTPException as e:
            # Создаём неудачный результат для этого изображения
            async def failed_result(idx: int, error_msg: str):
                return ClassifyResult(
                    task_id=f"{task_id}_{idx}",
                    success=False,
                    error=error_msg,
                    processing_time_ms=0
                )
            tasks.append(failed_result(i, e.detail))
    
    # Запускаем все задачи параллельно
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time_ms = (time.time() - start_time) * 1000
    
    responses = []
    for result in results:
        if isinstance(result, Exception):
            responses.append(ClassifyResponse(
                success=False,
                task_id="",
                error=str(result),
                processing_time_ms=0
            ))
        else:
            responses.append(ClassifyResponse(
                success=result.success,
                task_id=result.task_id,
                category=result.category,
                confidence=result.confidence,
                all_scores=result.all_scores or {},
                processing_time_ms=result.processing_time_ms,
                error=result.error
            ))
    
    return BatchClassifyResponse(
        success=True,
        task_id=task_id,
        results=responses,
        processing_time_ms=total_time_ms,
        error=None
    )

# ======================================================================
# Health & Info
# ======================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса."""
    if not worker or not worker.is_healthy():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "device": config.DEVICE,
                "model": config.MODEL_NAME,
                "queue_size": input_queue.qsize() if input_queue else 0,
                "tasks_processed": worker.tasks_processed if worker else 0
            }
        )
    
    return HealthResponse(
        status="healthy",
        device=config.DEVICE,
        model=config.MODEL_NAME,
        queue_size=input_queue.qsize(),
        tasks_processed=worker.tasks_processed
    )


@app.get("/info")
async def get_info():
    """Информация о сервисе."""
    logger.info(f"=== INFO CALLED ===")
    return {
        "service": "CLIP Classification Service",
        "version": "1.0.0",
        "model": config.MODEL_NAME,
        "device": config.DEVICE,
        "default_categories": config.DEFAULT_CATEGORIES,
        "max_images_per_batch": config.MAX_IMAGES_PER_BATCH,
        "max_image_size_mb": config.MAX_IMAGE_SIZE_MB,
        "queue_maxsize": config.QUEUE_MAXSIZE,
        "endpoints": [
            "/classify",
            "/classify/batch",
            "/validate",
            "/validate/batch",
            "/health",
            "/info"
        ]
    }


# ======================================================================
# Точка входа
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level=config.LOGGING_LEVEL.lower()
    )