import time
import logging
from fastapi import FastAPI, HTTPException
from src.domain.schemas import ProductRequest, ProductResponse
from src.services.scraper import WebScraper
from src.services.inference import InferenceService

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("FurnitureAPI")

# 1. App Initialization
app = FastAPI(
    title="Furniture NER API",
    description="Extract furniture products from e-commerce websites using DistilBERT",
    version="1.0.0"
)

# 2. Service Initialization
# We initialize the model only once at startup
try:
    inference_service = InferenceService()
    logger.info("Inference Service initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Inference Service: {e}")
    raise e

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Service is running. Go to /docs for Swagger UI."}

@app.post("/extract", response_model=ProductResponse)
def extract_products(request: ProductRequest):
    """
    Main endpoint:
    1. Scrapes the URL.
    2. Extracts products using the ML model.
    3. Returns structured data.
    """
    start_time = time.time()
    url_str = str(request.url)
    
    logger.info(f"Received request for URL: {url_str}")

    # Step 1: Scraping
    text = WebScraper.get_clean_text(url_str)
    
    if not text:
        logger.warning(f"Scraping failed for {url_str}")
        raise HTTPException(status_code=400, detail="Failed to retrieve content from the URL. The site might be blocking bots.")

    if len(text) < 50:
        logger.warning(f"Content too short for {url_str} (Length: {len(text)})")
        raise HTTPException(status_code=422, detail="Page content is too short or empty.")

    # Step 2: Inference
    try:
        products = inference_service.predict(text)
        logger.info(f"Extracted {len(products)} products from {url_str}")
    except Exception as e:
        logger.error(f"ML Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Model Error: {str(e)}")
    
    # Step 3: Reponse
    processing_time = time.time() - start_time
    
    return ProductResponse(
        url=url_str,
        products=products,
        product_count=len(products),
        processing_time=round(processing_time, 2)
    )

# Debug entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
    