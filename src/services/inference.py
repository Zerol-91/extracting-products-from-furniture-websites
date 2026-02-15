import re
from typing import List
from transformers import pipeline
from src.core.config import MODEL_PATH, CONFIDENCE_THRESHOLD, CHUNK_SIZE, OVERLAP

class InferenceService:
    """
    Singleton service responsible for loading the NER model and performing inference.
    """
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        print(f"Loading model from: {MODEL_PATH}...")
        try:
            self._pipeline = pipeline(
                "token-classification",
                model=MODEL_PATH,
                tokenizer=MODEL_PATH,
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model. {e}")
            self._pipeline = None

    def predict(self, text: str) -> List[str]:
        if not self._pipeline or not text:
            return []

        # Sliding window logic
        all_results = []
        for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
            chunk = text[i : i + CHUNK_SIZE]
            if len(chunk) < 10:
                continue
            try:
                chunk_results = self._pipeline(chunk)
                all_results.extend(chunk_results)
            except Exception:
                continue

        # Post-processing
        products = set()
        for res in all_results:
            if res['entity_group'] == 'PROD' and res['score'] > CONFIDENCE_THRESHOLD:
                word = res['word'].strip()
                
                word = word.replace("##", "")
                
                # Basic heuristics
                if len(word) < 3 or not any(c.isalpha() for c in word):
                    continue
                
                # Cleaning price artifacts
                word = re.sub(r'\$\d+.*', '', word).strip()
                
                # Clean punctuation
                word = word.strip(" .,-:")
                if len(word) > 2:
                    products.add(word) 

        return list(products)