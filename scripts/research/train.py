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
from scripts.research.data_processor import prepare_datasets, MODEL_CHECKPOINT, LABEL_LIST

def compute_metrics(p):
    """
    Calculates metrics (Precision, Recall, F1) during training.
    Uses the seqeval library (the standard for NER).
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Removing ignored tokens (-100)
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
    print("1. Preparing the data...")
    tokenized_datasets = prepare_datasets()

    print("\n--- SANITY CHECK DATA ---")
    labels = [item for sublist in tokenized_datasets["train"]["labels"] for item in sublist]
    b_prod_count = labels.count(1) # 1 = B-PROD
    i_prod_count = labels.count(2) # 2 = I-PROD
    total_tokens = len(labels)
    
    print(f"Total tokens: {total_tokens}")
    print(f"B-PROD tags (Product Start): {b_prod_count}")
    print(f"I-PROD tags (Continuation): {i_prod_count}")
    
    if b_prod_count == 0:
        print("🔴 CRITICAL ERROR: There are no products in the data! Check data_processor.py")
        return # Stop the script, there's no point in learning
    else:
        print(f"🟢 Data available. Product share: {((b_prod_count + i_prod_count) / total_tokens):.2%}")
    print("-----------------------------\n")
    
    
    print("2. Loading the model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(LABEL_LIST),
        id2label={i: l for i, l in enumerate(LABEL_LIST)},
        label2id={l: i for i, l in enumerate(LABEL_LIST)},
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # The tokenizer is transferred here
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Hyperparameters
    args = TrainingArguments(
        output_dir="models/checkpoint",
        eval_strategy="epoch",
        save_strategy="epoch",     
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.005,

        load_best_model_at_end=True, # load the epoch where the metric was best
        metric_for_best_model="f1",
        greater_is_better=True, 
        save_total_limit=1,

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("3. Start of training! (This may take 10-20 minutes)...")
    trainer.train()

    print("4. Saving the final model...")
    # We save the tokenizer manually, it's more reliable.
    model.save_pretrained("models/final_model")
    tokenizer.save_pretrained("models/final_model")
    print("Done! The model is saved in models/final_model")

if __name__ == "__main__":
    main()