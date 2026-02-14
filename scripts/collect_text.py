import os
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import json

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def clean_text(url):
    """Скачивает текст с жестким таймаутом."""
    try:
        # 1. Скачиваем HTML
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # 2. Парсим через BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. УДАЛЯЕМ МУСОР (Самая важная часть)
        # Удаляем скрипты, стили, навигацию и подвалы. 
        # Это оставит "грязноватый", но ПОЛНЫЙ текст с товарами.
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"]):
            tag.decompose() # Полностью вырезаем эти теги

        # 4. Достаем текст
        # separator='\n' важен, чтобы слова из разных блоков не слипались (PriceName -> Price Name)
        text = soup.get_text(separator="\n", strip=True)
        
        # 5. Чистим от лишних пустых строк (косметика)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        
        return clean_text

    except Exception as e:
        # print(f"Skipping {url}: {e}")
        return None

def main():
    # 1. Читаем URL
    # if not os.path.exists("data/raw/urls.txt"):
    #     print("Создай файл data/raw/urls.txt и положи туда ссылки!")
    #     return

    with open("data/raw/urls.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Берем 60-80 ссылок. Этого хватит для PoC.
    # Если ссылок мало - бери все.
    urls_to_process = urls[:400] 
    
    collected_data = []
    
    print(f"Скачиваем тексты с {len(urls_to_process)} сайтов...")

    for i, url in enumerate(tqdm(urls_to_process)):
        text = clean_text(url)
        
        if text and len(text) > 50:
            collected_data.append({
                "id": i,
                "url": url,
                "text": text,
                "products": [] # ЭТО ТЫ ЗАПОЛНИШЬ РУКАМИ
            })

    # Сохраняем в JSON (не JSONL, чтобы тебе было удобнее редактировать)
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/manual_dataset.json", "w", encoding="utf-8") as f:
        json.dump(collected_data, f, indent=4, ensure_ascii=False)
    
    print(f"Готово! Сохранено {len(collected_data)} текстов в data/processed/manual_dataset.json")
    print("Теперь открой этот файл и заполни поля 'products'.")

if __name__ == "__main__":
    main()