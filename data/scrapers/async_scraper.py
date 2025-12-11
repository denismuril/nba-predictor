import aiohttp
import asyncio
import logging
import random
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# User-Agent rotation pool para evitar bloqueios
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
]


class AsyncScraper:
    """
    Base class for asynchronous scraping with rate limiting and retries.
    Já possui retry logic built-in com exponential backoff.
    """
    def __init__(self, max_concurrent_requests: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._ua_index = 0

    def _get_headers(self, url: str = '') -> Dict[str, str]:
        """Get headers with rotating User-Agent, customized per site."""
        # Rotate User-Agent
        ua = USER_AGENTS[self._ua_index % len(USER_AGENTS)]
        self._ua_index += 1
        
        # Headers base
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        
        # Headers especiais para Basketball-Reference (mais exigente)
        if 'basketball-reference.com' in url:
            headers.update({
                'Referer': 'https://www.google.com/',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-User': '?1',
            })
        elif 'nba.com' in url:
            headers.update({
                'Referer': 'https://www.nba.com/',
                'Origin': 'https://www.nba.com',
            })
        
        return headers

    async def fetch_json(self, url: str, params: Optional[Dict] = None, retries: int = 3) -> Optional[Any]:
        """
        Fetch JSON data from a URL with retries and rate limiting.
        
        Built-in retry logic:
        - Attempt 1: imediato
        - Attempt 2: espera 2s
        - Attempt 3: espera 4s  
        """
        async with self.semaphore:
            # Random sleep to behave more like a human and avoid strict rate limits
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            for attempt in range(retries):
                try:
                    headers = self._get_headers(url)
                    async with aiohttp.ClientSession(headers=headers) as session:
                        async with session.get(url, params=params, timeout=15) as response:
                            if response.status == 200:
                                return await response.json()
                            elif response.status in (429, 403):
                                # Rate limit - exponential backoff extra
                                wait_time = (2 ** attempt) + random.uniform(0, 1)
                                logger.warning(f"⚠️ Rate limit hit for {url}. Waiting {wait_time:.2f}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"❌ Error {response.status} fetching {url}")
                                return None
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = (2 ** attempt)
                        logger.warning(f"⚠️ Connection error for {url}: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ Failed to fetch {url} after {retries} attempts: {e}")
                        return None
            return None

    async def fetch_text(self, url: str, retries: int = 3) -> Optional[str]:
        """
        Fetch HTML/Text content from a URL.
        """
        async with self.semaphore:
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            for attempt in range(retries):
                try:
                    headers = self._get_headers(url)
                    timeout = 30 if 'basketball-reference' in url else 15
                    async with aiohttp.ClientSession(headers=headers) as session:
                        async with session.get(url, timeout=timeout) as response:
                            if response.status == 200:
                                return await response.text()
                            elif response.status in (429, 403):
                                # Rate limit or blocked - retry with backoff
                                wait_time = (2 ** attempt) + random.uniform(1, 3)
                                logger.warning(f"⚠️ {response.status} for {url}. Waiting {wait_time:.1f}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"❌ Error {response.status} fetching {url}")
                                return None
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = (2 ** attempt)
                        logger.warning(f"⚠️ Retry {attempt+1}/{retries} for {url} after error: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ Failed to fetch {url}: {e}")
                        return None
            return None
