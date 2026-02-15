import json
import random
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Constants for label mapping
# O = Outside (not a product), B = Beginning, I = Inside (continuation)
LABEL_LIST = ["O", "B-PROD", "I-PROD"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Use "DistilBERT" - it is fast and easy
MODEL_CHECKPOINT = "distilbert-base-uncased" 

def load_raw_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out empty ones or those where there are no products (for training NER we need examples with entities
    return [x for x in data if x['text']]

def find_span_indices(text, product_name):
    """
    Finds the beginning and end of the substring product_name in text.
    Returns (start_char, end_char) or None.
    """
    start_idx = text.find(product_name)
    if start_idx == -1:
        return None
    return start_idx, start_idx + len(product_name)

def create_bio_tags(example):
    """
    Converts text and product lists into a list of tags for each word.
    This is a simplified whitespace tokenization for pre-tagging.
    """
    text = example['text']
    products = example['products']
    
    # First, we create a list of character labels (0 - nothing, 1 - start, 2 - continue)
    # These are "Character-level masks"
    char_labels = [0] * len(text)
    
    for prod in products:
        clean_prod = prod.strip()
        if not clean_prod: 
            continue
            
        span = find_span_indices(text, clean_prod)
        if span:
            start, end = span
            # B-tag for first character 
            char_labels[start] = 1 
            # I-tag for the rest
            for i in range(start + 1, end):
                char_labels[i] = 2
    
    return {"text": text, "char_labels": char_labels, "orig_products": products}

def tokenize_and_align_labels(examples):
    """
    Migrating char_labels to BERT tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512, # Cutting down long websites
        is_split_into_words=False, # submit the raw text
        return_offsets_mapping=True # Returns the tokens' position in the source text.
    )

    labels = []
    
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        char_label = examples["char_labels"][i]
        doc_labels = []
        
        for start_char, end_char in offsets:
            # Special tokens ([CLS], [SEP]) have offset (0, 0)
            if start_char == end_char == 0:
                doc_labels.append(-100) # -100 ignored when calculating Loss
                continue
            
            # If at least one character within the token is marked as a product -> it is a product
            # Take the label from the middle of the token or from the beginning
            token_label = 0
            
            # Checking if the token matches the entity
            # If the beginning of the token coincides with the beginning of the entity (1) -> B-PROD
            if char_label[start_char] == 1:
                token_label = LABEL2ID["B-PROD"]
            # If inside the entity (2) -> I-PROD
            elif char_label[start_char] == 2:
                token_label = LABEL2ID["I-PROD"]
            
            doc_labels.append(token_label)
            
        labels.append(doc_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_datasets(json_path="data/processed/manual_dataset.json"):
    raw_data = load_raw_data(json_path)
    
    # 1. Adding BIO tags at the character level
    processed_data = [create_bio_tags(x) for x in raw_data]
    
    # 2. converting into the HF Dataset
    hf_dataset = Dataset.from_list(processed_data)
    
    # 3. splitting Train / Test (85% / 15%)
    # seed=42 for reproducibility
    split_dataset = hf_dataset.train_test_split(test_size=0.15, seed=42)
    
    print(f"Train size: {len(split_dataset['train'])}")
    print(f"Test size: {len(split_dataset['test'])}")
    
    # 4. Tokenization and alignment 
    # batched=True speeds up the process
    tokenized_datasets = split_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        remove_columns=["text", "char_labels", "orig_products"] # remove raw data, leaving tensors
    )
    
    return tokenized_datasets

if __name__ == "__main__":
    # Test run
    ds = prepare_datasets()
    print("Example of tokens:", ds['train'][0]['input_ids'][:10])
    print("Example of labels:", ds['train'][0]['labels'][:10])
    print("Success! The data is ready to be fed to BERT.")