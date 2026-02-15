import json
import difflib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.services.inference import InferenceService

def load_test_data(filepath="data/processed/manual_dataset.json"):
    """We load the data and take ONLY the test portion (the same 15% as during training)"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter empty ones as training
    data = [x for x in data if x['text']]
    
    # random_state=42 must match train.py to ensure the same data
    _, test_data = train_test_split(data, test_size=0.15, random_state=42)
    return test_data

def is_soft_match(pred, truth, threshold=0.6):
    """
    Checks if strings are similar.
    1. If one is inside the other (Substring) -> True
    2. If the character similarity is > 60% (Levenshtein) -> True
    """
    pred_clean = pred.lower().strip()
    truth_clean = truth.lower().strip()
    
    # Checking for a substring
    if pred_clean in truth_clean or truth_clean in pred_clean:
        return True
        
    # Check for similarities (typos, extra words)
    similarity = difflib.SequenceMatcher(None, pred_clean, truth_clean).ratio()
    return similarity >= threshold

def calculate_soft_metrics(extractor, test_data):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print(f"Running validation on {len(test_data)} examples...\n")

    for item in tqdm(test_data):
        text = item['text']
        true_products = set(item['products']) # True goods
        
        # Model Prediction
        pred_products = set(extractor.predict(text))
        
        # Calculating metrics for one document
        # create copies of sets to delete found ones
        local_tp = 0
        unmatched_true = list(true_products)
        
        # checks every prediction
        for pred in pred_products:
            match_found = False
            for true_prod in unmatched_true:
                if is_soft_match(pred, true_prod):
                    match_found = True
                    unmatched_true.remove(true_prod) # removes it so as not to count twice
                    break
            
            if match_found:
                local_tp += 1
            else:
                total_fp += 1 # predicted it, but it's not in the markup.
        
        total_tp += local_tp
        total_fn += len(unmatched_true) # What remains undiscovered

    # Final metrics
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
    # Loading the model
    print("Загрузка модели...")
    extractor = InferenceService()
    
    # Loading the data
    test_data = load_test_data()
    
    # counts
    metrics = calculate_soft_metrics(extractor, test_data)
    
    print("\n" + "="*30)
    print("📊 RESULTS SOFT MATCHING")
    print("="*30)
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:     {metrics['recall']:.2%}")
    print(f"F1 Score (Soft):      {metrics['f1']:.2%}")
    print("-" * 30)
    print(f"Found correctly (TP):   {metrics['tp']}")
    print(f"Excessive noise (FP):      {metrics['fp']}")
    print(f"Missed (FN):       {metrics['fn']}")
    print("="*30)