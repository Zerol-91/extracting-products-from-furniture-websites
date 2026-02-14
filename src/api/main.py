import time
from fastapi import FastAPI, HTTPException
from src.domain.schemas import ProductRequest, ProductResponse
from src.services.scraper import WebScraper
from src.services.inference import InferenceService

# 1. Инициализация приложения
app = FastAPI(
    title="Furniture NER API",
    description="Extract furniture products from e-commerce websites using DistilBERT",
    version="1.0.0"
)

# 2. Инициализация сервисов (Model Loading)
# Мы создаем экземпляр сервиса при старте, чтобы модель загрузилась в память сразу.
inference_service = InferenceService()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Service is running. Go to /docs for Swagger UI."}

@app.post("/extract", response_model=ProductResponse)
async def extract_products(request: ProductRequest):
    """
    Main endpoint:
    1. Scrapes the URL.
    2. Extracts products using the ML model.
    3. Returns structured data.
    """
    start_time = time.time()
    url_str = str(request.url)
    
    print(f"Processing URL: {url_str}")

    # Шаг 1: Скачивание (Scraping)
    text = WebScraper.get_clean_text(url_str)
    
    if not text:
        raise HTTPException(status_code=400, detail="Failed to retrieve content from the URL. The site might be blocking bots.")

    if len(text) < 50:
        raise HTTPException(status_code=422, detail="Page content is too short or empty.")

    # Шаг 2: Инференс (ML Prediction)
    try:
        products = inference_service.predict(text)
    except Exception as e:
        print(f"ML Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Model Error")

    # Шаг 3: Формирование ответа
    processing_time = time.time() - start_time
    
    return ProductResponse(
        url=url_str,
        products=products,
        product_count=len(products),
        processing_time=round(processing_time, 2)
    )

# Для отладки (запуск файла напрямую)
if __name__ == "__main__":
    import uvicorn
    # Запуск сервера на порту 8000
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)