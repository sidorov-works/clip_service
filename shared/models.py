# shared/models.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from dataclasses import dataclass


class ClassifyRequest(BaseModel):
    """Запрос на классификацию одного изображения."""
    image: str = Field(..., description="Base64-строка изображения (с префиксом data:image/... или без)")
    categories: Optional[List[str]] = Field(None, description="Категории для классификации (опционально)")


class ClassifyResponse(BaseModel):
    """Ответ на запрос классификации одного изображения."""
    success: bool
    task_id: str
    category: str = ""
    confidence: float = 0.0
    all_scores: Dict[str, float] = Field(default_factory=dict)
    processing_time_ms: float = 0
    error: Optional[str] = None


class BatchClassifyRequest(BaseModel):
    """Батчевый запрос на классификацию нескольких изображений."""
    images: List[str] = Field(..., description="Список base64-строк изображений")
    categories: Optional[List[str]] = Field(None, description="Категории для классификации (опционально)")


class BatchClassifyResponse(BaseModel):
    """Батчевый ответ."""
    success: bool
    task_id: str
    results: List[ClassifyResponse]
    processing_time_ms: float = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    model: str
    queue_size: int
    tasks_processed: int


# --- Внутренние модели для очереди ---

@dataclass
class ClassifyTask:
    """Задача для воркера (одно изображение)."""
    task_id: str
    image_bytes: bytes
    categories: List[str]
    created_at: float


@dataclass
class ClassifyResult:
    """Результат классификации для одного изображения."""
    task_id: str
    success: bool
    category: str = ""
    confidence: float = 0.0
    all_scores: Dict[str, float] = None
    error: str = None
    processing_time_ms: float = 0