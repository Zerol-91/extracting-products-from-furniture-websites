from transformers import pipeline

class ProductExtractor:
    def __init__(self, model_path="models/final_model"):
        """
        Инициализируем пайплайн.
        aggregation_strategy="simple" - это магия, которая сама склеивает 
        B-PROD и I-PROD обратно в целые слова.
        """
        try:
            self.pipe = pipeline(
                "token-classification", 
                model=model_path, 
                tokenizer=model_path,
                aggregation_strategy="simple", # Важно! Склеивает токены в слова
                device=-1 # -1 для CPU, 0 для GPU
            )
            print(f"✅ Модель загружена из {model_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.pipe = None

    def predict(self, text):
        if not self.pipe:
            return []
            
        chunk_size = 2000  # Размер куска (символы)
        overlap = 200      # Перекрытие, чтобы не разрезать слово "Chair" пополам
        
        all_results = []
        
        # Цикл по тексту с шагом (chunk_size - overlap)
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            
            # Если кусок слишком короткий, пропускаем
            if len(chunk) < 10:
                continue
                
            try:
                # Прогоняем кусок через модель
                chunk_results = self.pipe(chunk)
                
                # Добавляем результаты, корректируя их, но нам важен сам текст сущности
                all_results.extend(chunk_results)
            except Exception as e:
                # Иногда бывает ошибка токенизации на странных символах
                # print(f"Warning on chunk: {e}") 
                continue

        # --- ФИЛЬТРАЦИЯ И СБОРКА ---
        products = []
        for res in all_results:
            # entity_group - это название класса (PROD), score - уверенность
            if res['entity_group'] == 'PROD' and res['score'] > 0.10: # Порог уверенности 40%
                word = res['word'].strip()
                # Фильтруем совсем мусор (одной буквой или символами)
                if len(word) > 2 and any(c.isalpha() for c in word):
                    products.append(word)
        
        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        unique_products = []
        for p in products:
            if p not in seen:
                unique_products.append(p)
                seen.add(p)
                
        return unique_products

# Тест (запустится только если выполнить файл напрямую)
if __name__ == "__main__":
    # Эмуляция работы
    extractor = ProductExtractor()
    text = "Header... " + "blah " * 500 + "Buy our new Super Sofa 3000 down here!"
    print(extractor.predict(text))