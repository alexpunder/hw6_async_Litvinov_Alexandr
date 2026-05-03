import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_NAME = "cointegrated/rubert-tiny2"

class ReloaderPredictor:
    def __init__(self, loading_pattern):
        self.model_name = MODEL_NAME
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._loading_pattern=loading_pattern

    def _load_components(self):
        self._loading_pattern()

    def _clear_components(self):
        print("[Принудительная перезагрузка] Очистка модели из памяти...")
        del self._pipeline
        del self._model
        del self._tokenizer
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[Принудительная перезагрузка] Модель очищена.")

    def predict_and_clear(self, text):
        self._load_components()
        result = self._pipeline(text)
        self._clear_components()
        return result

def lazy():                                                                         # --- 1. Lazy Loading Pattern ---
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _pipeline = pipeline("sentiment-analysis", model=_model, tokenizer=_tokenizer)


reloader_predictor=ReloaderPredictor(lazy())

start_time = time.time()
result = lazy()
print(f"[Ленивая загрузка] Результат первого предсказания: {result} (заняло {time.time() - start_time:.2f}с)")

start_time = time.time()
result = reloader_predictor._clear_components()
fs=time.time() - start_time
print(f"[Ленивая загрузка] Результат второго предсказания: {result} (заняло {fs:.2f}с)")
