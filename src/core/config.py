import os
from pathlib import Path

# Define base paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model")

# Inference settings
CONFIDENCE_THRESHOLD = 0.10
CHUNK_SIZE = 2000
OVERLAP = 200