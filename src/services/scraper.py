import requests
import urllib3
from bs4 import BeautifulSoup
from typing import Optional
import logging


# Suppress warnings about unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class WebScraper:
    """
    Service responsible for fetching and cleaning HTML content.
    """
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    @staticmethod
    def get_clean_text(url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=WebScraper.HEADERS, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove garbage tags
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg", "button", "input"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            
            # Remove empty lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)

        except requests.exceptions.SSLError as e:
            logger.error(f"SSL Error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            return None