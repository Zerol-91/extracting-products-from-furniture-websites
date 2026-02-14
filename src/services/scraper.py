import requests
from bs4 import BeautifulSoup
from typing import Optional

class WebScraper:
    """
    Service responsible for fetching and cleaning HTML content.
    """
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    @staticmethod
    def get_clean_text(url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=WebScraper.HEADERS, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove garbage tags
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            
            # Remove empty lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)

        except Exception as e:
            print(f"Scraping error for {url}: {e}")
            return None