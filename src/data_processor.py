import json
import random
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Константы для маппинга лейблов
# O = Outside (не товар), B = Beginning (начало), I = Inside (продолжение)
LABEL_LIST = ["O", "B-PROD", "I-PROD"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Используем "DistilRoBERTa" или "DistilBERT" - они быстрые и легкие
# RoBERTa часто лучше работает с "грязным" вебом
MODEL_CHECKPOINT = "distilbert-base-uncased" 

def load_raw_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Фильтруем пустые или те, где нет продуктов (для обучения NER нам нужны примеры с сущностями, 
    # хотя пустые тоже полезны для negative sampling, но пока возьмем 80% с сущностями)
    return [x for x in data if x['text']]

def find_span_indices(text, product_name):
    """
    Ищет начало и конец подстроки product_name в text.
    Возвращает (start_char, end_char) или None.
    """
    start_idx = text.find(product_name)
    if start_idx == -1:
        return None
    return start_idx, start_idx + len(product_name)

def create_bio_tags(example):
    """
    Превращает текст и список продуктов в список тегов для каждого слова.
    Это упрощенная токенизация по пробелам для предварительной разметки.
    """
    text = example['text']
    products = example['products']
    
    # Сначала создаем список символьных меток (0 - ничего, 1 - начало, 2 - продолжение)
    # Это "Character-level masks"
    char_labels = [0] * len(text)
    
    for prod in products:
        # Важно: strip() убирает лишние пробелы, которые могут мешать поиску
        clean_prod = prod.strip()
        if not clean_prod: 
            continue
            
        span = find_span_indices(text, clean_prod)
        if span:
            start, end = span
            # B-tag для первого символа
            char_labels[start] = 1 
            # I-tag для остальных
            for i in range(start + 1, end):
                char_labels[i] = 2
    
    return {"text": text, "char_labels": char_labels, "orig_products": products}

def tokenize_and_align_labels(examples):
    """
    Перенос char_labels на токены BERT.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512, # Обрезаем длинные сайты, BERT больше не ест
        is_split_into_words=False, # Мы подаем сырой текст
        return_offsets_mapping=True # Важно! Возвращает позиции токенов в исходном тексте
    )

    labels = []
    
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        char_label = examples["char_labels"][i]
        doc_labels = []
        
        for start_char, end_char in offsets:
            # Спецтокены ([CLS], [SEP]) имеют offset (0, 0)
            if start_char == end_char == 0:
                doc_labels.append(-100) # -100 игнорируется при расчете Loss
                continue
            
            # Если хотя бы один символ внутри токена помечен как товар -> это товар
            # Берем метку из середины токена или начала
            token_label = 0
            
            # Проверяем, попадает ли токен на сущность
            # Если начало токена совпадает с началом сущности (1) -> B-PROD
            if char_label[start_char] == 1:
                token_label = LABEL2ID["B-PROD"]
            # Если внутри сущности (2) -> I-PROD
            elif char_label[start_char] == 2:
                token_label = LABEL2ID["I-PROD"]
            
            doc_labels.append(token_label)
            
        labels.append(doc_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_datasets(json_path="data/processed/manual_dataset.json"):
    raw_data = load_raw_data(json_path)
    
    # 1. Добавляем BIO теги на уровне символов
    processed_data = [create_bio_tags(x) for x in raw_data]
    
    # 2. Конвертируем в HF Dataset
    hf_dataset = Dataset.from_list(processed_data)
    
    # 3. Делим на Train / Test (85% / 15%)
    # seed=42 для воспроизводимости
    split_dataset = hf_dataset.train_test_split(test_size=0.15, seed=42)
    
    print(f"Train size: {len(split_dataset['train'])}")
    print(f"Test size: {len(split_dataset['test'])}")
    
    # 4. Токенизация и выравнивание (Map - это быстро)
    # batched=True ускоряет процесс
    tokenized_datasets = split_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        remove_columns=["text", "char_labels", "orig_products"] # Удаляем сырые данные, оставляем тензоры
    )
    
    return tokenized_datasets

if __name__ == "__main__":
    # Тестовый запуск
    ds = prepare_datasets()
    print("Пример токенов:", ds['train'][0]['input_ids'][:10])
    print("Пример лейблов:", ds['train'][0]['labels'][:10])
    print("Успех! Данные готовы к скармливанию BERTу.")