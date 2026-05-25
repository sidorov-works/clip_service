# shared/models.py
"""
Pydantic модели для API и dataclass'ы для внутренней очереди.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from dataclasses import dataclass


# ======================================================================
# Pydantic модели для API

# --- Классификация (одиночная) ---

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


# --- Классификация (батчевая) ---

class BatchClassifyRequest(BaseModel):
    """Батчевый запрос на классификацию нескольких изображений."""
    images: List[str] = Field(..., description="Список base64-строк изображений")
    categories: Optional[List[str]] = Field(None, description="Категории для классификации (опционально)")


class BatchClassifyResponse(BaseModel):
    """Батчевый ответ классификации."""
    success: bool
    task_id: str
    results: List[ClassifyResponse]
    processing_time_ms: float = 0
    error: Optional[str] = None


# --- Валидация OCR (одиночная) ---

class ValidateRequest(BaseModel):
    """Запрос на валидацию OCR результата."""
    image: str = Field(..., description="Base64-строка изображения")
    text: str = Field(..., description="Распознанный текст")
    threshold: Optional[float] = Field(0.5, description="Порог уверенности (0-1)")


class ValidateResponse(BaseModel):
    """Ответ валидации OCR."""
    success: bool
    task_id: str
    is_valid: bool = False          # True если текст соответствует изображению
    confidence: float = 0.0         # Сходство между изображением и текстом (0-1)
    processing_time_ms: float = 0
    error: Optional[str] = None


# --- Валидация OCR (батчевая) ---

class BatchValidateRequest(BaseModel):
    """Батчевый запрос на валидацию OCR результатов."""
    images: List[str] = Field(..., description="Список base64-строк изображений")
    texts: List[str] = Field(..., description="Список распознанных текстов (позиция соответствует изображению)")
    threshold: Optional[float] = Field(0.5, description="Порог уверенности")


class BatchValidateResponse(BaseModel):
    """Батчевый ответ валидации."""
    success: bool
    task_id: str
    results: List[ValidateResponse]
    processing_time_ms: float = 0
    error: Optional[str] = None


# --- Health & Info ---

class HealthResponse(BaseModel):
    """Ответ health check."""
    status: str
    device: str
    model: str
    queue_size: int
    tasks_processed: int


# ======================================================================
# Внутренние модели для очереди (dataclass, не Pydantic)

@dataclass
class ClassifyTask:
    """Задача для классификации (одно изображение)."""
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


@dataclass
class ValidateTask:
    """Задача для валидации OCR (изображение + текст)."""
    task_id: str
    image_bytes: bytes
    text: str
    created_at: float


@dataclass
class ValidateResult:
    """Результат валидации для одной пары (изображение, текст)."""
    task_id: str
    success: bool
    confidence: float = 0.0
    error: str = None
    processing_time_ms: float = 0