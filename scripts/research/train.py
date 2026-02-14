import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer
)
from src.data_processor import prepare_datasets, MODEL_CHECKPOINT, LABEL_LIST

def compute_metrics(p):
    """
    Вычисляет метрики (Precision, Recall, F1) во время обучения.
    Использует библиотеку seqeval (стандарт для NER).
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Убираем игнорируемые токены (-100)
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    print("1. Готовим данные...")
    tokenized_datasets = prepare_datasets()

    print("\n--- SANITY CHECK ДАННЫХ ---")
    labels = [item for sublist in tokenized_datasets["train"]["labels"] for item in sublist]
    b_prod_count = labels.count(1) # 1 = B-PROD
    i_prod_count = labels.count(2) # 2 = I-PROD
    total_tokens = len(labels)
    
    print(f"Всего токенов: {total_tokens}")
    print(f"Тегов B-PROD (Начало товара): {b_prod_count}")
    print(f"Тегов I-PROD (Продолжение): {i_prod_count}")
    
    if b_prod_count == 0:
        print("🔴 КРИТИЧЕСКАЯ ОШИБКА: В данных нет ни одного товара! Проверь data_processor.py")
        return # Останавливаем скрипт, нет смысла учить
    else:
        print(f"🟢 Данные есть. Доля товаров: {((b_prod_count + i_prod_count) / total_tokens):.2%}")
    print("-----------------------------\n")
    
    
    print("2. Загружаем модель...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(LABEL_LIST),
        id2label={i: l for i, l in enumerate(LABEL_LIST)},
        label2id={l: i for i, l in enumerate(LABEL_LIST)},
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # Токенизатор передается СЮДА, этого достаточно для работы с данными
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Гиперпараметры
    args = TrainingArguments(
        output_dir="models/checkpoint",
        eval_strategy="epoch",       # <--- Исправление №1 (новое название аргумента)
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.005,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # tokenizer=tokenizer,      # <--- Исправление №2: УДАЛИЛИ ЭТУ СТРОКУ (она вызывает ошибку)
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("3. Начинаем обучение! (Может занять 10-20 минут)...")
    trainer.train()

    print("4. Сохраняем финальную модель...")
    # Мы сохраняем токенизатор вручную, так надежнее
    model.save_pretrained("models/final_model")
    tokenizer.save_pretrained("models/final_model")
    print("Готово! Модель сохранена в models/final_model")

if __name__ == "__main__":
    main()