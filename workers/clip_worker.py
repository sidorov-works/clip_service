# workers/clip_worker.py
import asyncio
from shared.config import config
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import io
import torch

from shared.models import ClassifyTask, ClassifyResult


class CLIPWorker:
    """
    Единственный владелец модели CLIP.
    Обрабатывает задачи из очереди последовательно.
    """
    
    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.model = None
        self.processor = None
        self.device = config.DEVICE
        self.tasks_processed = 0
    
    async def load_model(self):
        """Загружает модель CLIP (вызывается один раз при старте)."""
        def _load():
            model = CLIPModel.from_pretrained(config.MODEL_NAME)
            processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
            model = model.to(self.device)
            model.eval()
            return model, processor
        
        # ✅ run_in_executor для загрузки модели (тяжёлая операция)
        loop = asyncio.get_event_loop()
        self.model, self.processor = await loop.run_in_executor(None, _load)
        print(f"CLIP модель загружена на {self.device}")
    
    async def start(self):
        """Запускает цикл обработки задач."""
        await self.load_model()
        
        while self.running:
            try:
                task = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)
                result = await self._process_task(task)
                await self.output_queue.put(result)
                self.input_queue.task_done()
                self.tasks_processed += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Ошибка в воркере: {e}")
    
    async def _process_task(self, task: ClassifyTask) -> ClassifyResult:
        """Обрабатывает одну задачу (одно изображение)."""
        start_time = time.time()
        
        try:
            # Декодируем изображение
            image = Image.open(io.BytesIO(task.image_bytes)).convert("RGB")
            
            def _classify():
                inputs = self.processor(
                    text=task.categories,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0]
                
                best_idx = probs.argmax().item()
                all_scores = {
                    cat: score.item() 
                    for cat, score in zip(task.categories, probs)
                }
                
                return {
                    "category": task.categories[best_idx],
                    "confidence": probs[best_idx].item(),
                    "all_scores": all_scores
                }
            
            # ✅ asyncio.to_thread вместо явного получения event_loop
            result = await asyncio.to_thread(_classify)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return ClassifyResult(
                task_id=task.task_id,
                success=True,
                category=result["category"],
                confidence=result["confidence"],
                all_scores=result["all_scores"],
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return ClassifyResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )
    
    def is_healthy(self) -> bool:
        return self.model is not None
    
    async def stop(self):
        self.running = False