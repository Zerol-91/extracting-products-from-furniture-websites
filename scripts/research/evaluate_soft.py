import json
import difflib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from services.inference import ProductExtractor

def load_test_data(filepath="data/processed/manual_dataset.json"):
    """Загружаем данные и берем ТОЛЬКО тестовую часть (те же 15%, что при обучении)"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Фильтруем пустые, как делали в обучении
    data = [x for x in data if x['text']]
    
    # Важно! random_state=42 должен совпадать с train.py, чтобы данные были те же
    _, test_data = train_test_split(data, test_size=0.15, random_state=42)
    return test_data

def is_soft_match(pred, truth, threshold=0.6):
    """
    Проверяет, похожи ли строки.
    1. Если одна внутри другой (Substring) -> True
    2. Если сходство символов > 60% (Levenshtein) -> True
    """
    pred_clean = pred.lower().strip()
    truth_clean = truth.lower().strip()
    
    # Проверка на подстроку
    if pred_clean in truth_clean or truth_clean in pred_clean:
        return True
        
    # Проверка на похожесть (опечатки, лишние слова)
    similarity = difflib.SequenceMatcher(None, pred_clean, truth_clean).ratio()
    return similarity >= threshold

def calculate_soft_metrics(extractor, test_data):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print(f"Запуск валидации на {len(test_data)} примерах...\n")

    for item in tqdm(test_data):
        text = item['text']
        true_products = set(item['products']) # Истинные товары
        
        # Предсказание модели
        pred_products = set(extractor.predict(text))
        
        # Считаем метрики для одного документа
        # Создаем копии сетов, чтобы удалять найденное
        local_tp = 0
        unmatched_true = list(true_products)
        
        # Проверяем каждое предсказание
        for pred in pred_products:
            match_found = False
            for true_prod in unmatched_true:
                if is_soft_match(pred, true_prod):
                    match_found = True
                    unmatched_true.remove(true_prod) # Убираем, чтобы не посчитать дважды
                    break
            
            if match_found:
                local_tp += 1
            else:
                total_fp += 1 # Предсказал, но такого нет в разметке
        
        total_tp += local_tp
        total_fn += len(unmatched_true) # То, что осталось ненайденным

    # Итоговые метрики
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

if __name__ == "__main__":
    # Загружаем модель
    print("Загрузка модели...")
    extractor = ProductExtractor()
    
    # Загружаем данные
    test_data = load_test_data()
    
    # Считаем
    metrics = calculate_soft_metrics(extractor, test_data)
    
    print("\n" + "="*30)
    print("📊 РЕЗУЛЬТАТЫ SOFT MATCHING")
    print("="*30)
    print(f"Precision (Точность): {metrics['precision']:.2%}")
    print(f"Recall (Полнота):     {metrics['recall']:.2%}")
    print(f"F1 Score (Soft):      {metrics['f1']:.2%}")
    print("-" * 30)
    print(f"Найдено верно (TP):   {metrics['tp']}")
    print(f"Лишний шум (FP):      {metrics['fp']}")
    print(f"Пропущено (FN):       {metrics['fn']}")
    print("="*30)