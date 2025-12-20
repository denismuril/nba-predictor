"""
Stealth Browser - Módulo anti-detecção para scrapers NBA.

Este módulo fornece funcionalidade base para todos os scrapers que precisam
evadir detecção de bots. Inclui:
- Playwright stealth patches
- User-agents realistas randomizados
- Geo-headers e impressões digitais TLS
- Simulação de comportamento humano (scroll, delays, movimentos)

v26.3: Módulo centralizado de anti-bot para todos os scrapers de props.
"""

import asyncio
import random
import logging
from typing import Optional, Tuple
from contextlib import asynccontextmanager

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)


# User-Agents realistas e modernos (atualizados 2024)
USER_AGENTS = [
    # Chrome Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Chrome Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# Viewports comuns (baseado em estatísticas reais)
VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 2560, "height": 1440},
]

# Timezones americanos (onde NBA é popular)
TIMEZONES = [
    "America/New_York",
    "America/Chicago", 
    "America/Los_Angeles",
    "America/Denver",
]

# Locales
LOCALES = ["en-US", "en-GB", "pt-BR"]


def get_random_user_agent() -> str:
    """Retorna user-agent aleatório realista."""
    return random.choice(USER_AGENTS)


def get_random_viewport() -> dict:
    """Retorna viewport aleatório comum."""
    return random.choice(VIEWPORTS)


def get_random_timezone() -> str:
    """Retorna timezone aleatório americano."""
    return random.choice(TIMEZONES)


async def human_delay(min_sec: float = 0.5, max_sec: float = 2.0):
    """
    Aplica delay aleatório com distribuição gaussiana (mais humano).
    
    Args:
        min_sec: Delay mínimo
        max_sec: Delay máximo
    """
    mean = (min_sec + max_sec) / 2
    std = (max_sec - min_sec) / 4
    delay = max(min_sec, min(random.gauss(mean, std), max_sec))
    await asyncio.sleep(delay)


async def human_scroll(page: Page, direction: str = "down"):
    """
    Realiza scroll suave como humano.
    
    Args:
        page: Página do Playwright
        direction: 'down', 'up', ou 'random'
    """
    if direction == "random":
        direction = random.choice(["down", "up"])
    
    viewport_height = await page.evaluate("window.innerHeight")
    scroll_amount = random.randint(int(viewport_height * 0.3), int(viewport_height * 0.7))
    
    if direction == "up":
        scroll_amount = -scroll_amount
    
    # Scroll com easing
    await page.evaluate(f"""
        () => {{
            const start = window.scrollY;
            const end = start + {scroll_amount};
            const duration = {random.randint(300, 600)};
            const startTime = performance.now();
            
            function easeOutQuad(t) {{
                return t * (2 - t);
            }}
            
            function scroll() {{
                const elapsed = performance.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                window.scrollTo(0, start + (end - start) * easeOutQuad(progress));
                if (progress < 1) requestAnimationFrame(scroll);
            }}
            
            requestAnimationFrame(scroll);
        }}
    """)
    
    await human_delay(0.2, 0.5)


async def human_mouse_move(page: Page, x: int, y: int):
    """
    Move mouse de forma não-linear (curva de Bezier).
    
    Args:
        page: Página do Playwright
        x: Posição X destino
        y: Posição Y destino
    """
    # Get current position
    current = await page.evaluate("() => ({ x: window.mouseX || 0, y: window.mouseY || 0 })")
    curr_x, curr_y = current.get('x', 0), current.get('y', 0)
    
    # Generate Bezier control points
    ctrl1_x = curr_x + random.randint(-50, 50)
    ctrl1_y = curr_y + random.randint(-50, 50)
    ctrl2_x = x + random.randint(-50, 50)
    ctrl2_y = y + random.randint(-50, 50)
    
    # Move in steps along curve
    steps = random.randint(10, 20)
    for i in range(steps + 1):
        t = i / steps
        # Cubic Bezier formula
        new_x = int((1-t)**3 * curr_x + 3*(1-t)**2*t*ctrl1_x + 3*(1-t)*t**2*ctrl2_x + t**3*x)
        new_y = int((1-t)**3 * curr_y + 3*(1-t)**2*t*ctrl1_y + 3*(1-t)*t**2*ctrl2_y + t**3*y)
        
        await page.mouse.move(new_x, new_y)
        await asyncio.sleep(random.uniform(0.01, 0.03))


async def apply_stealth_patches(page: Page):
    """
    Aplica patches JavaScript para evadir detecção.
    
    Args:
        page: Página do Playwright
    """
    # Ocultar webdriver
    await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    """)
    
    # Falsificar plugins
    await page.add_init_script("""
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                {0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"}},
                {0: {type: "application/pdf", suffixes: "pdf", description: "Portable Document Format"}}
            ]
        });
    """)
    
    # Falsificar linguagens
    await page.add_init_script("""
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en', 'pt-BR']
        });
    """)
    
    # Ocultar automação Chrome
    await page.add_init_script("""
        window.chrome = {
            runtime: {},
            loadTimes: function() {},
            csi: function() {},
            app: {}
        };
    """)
    
    # Prevenir detecção de headless
    await page.add_init_script("""
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => 8
        });
    """)
    
    await page.add_init_script("""
        Object.defineProperty(navigator, 'deviceMemory', {
            get: () => 8
        });
    """)


@asynccontextmanager
async def create_stealth_browser(
    headless: bool = True,
    proxy: Optional[str] = None,
    use_proxy_manager: bool = True
):
    """
    Context manager para criar browser stealth.
    
    Args:
        headless: Modo headless
        proxy: URL de proxy (opcional)
        use_proxy_manager: Se True, usa ProxyManager se proxy for None
        
    Yields:
        Tupla (browser, context, page)
        
    Exemplo:
        async with create_stealth_browser() as (browser, context, page):
            await page.goto("https://example.com")
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright não instalado. Execute: pip install playwright && playwright install chromium")
    
    # Integração com ProxyManager
    if proxy is None and use_proxy_manager:
        try:
            # Import tardio para evitar ciclo
            from infrastructure.proxy_manager import get_proxy_manager
            pm = get_proxy_manager()
            proxy = pm.get_proxy()
            if proxy:
                logger.debug(f"🔄 Proxy obtido do manager: {proxy}")
        except ImportError:
            pass  # ProxyManager não disponível ou erro de import
    
    browser = None
    try:
        async with async_playwright() as p:
            # Browser args para anti-detecção
            browser_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-infobars",
                "--window-position=0,0",
                "--ignore-certifcate-errors",
                "--ignore-certifcate-errors-spki-list",
            ]
            
            launch_opts = {
                "headless": headless,
                "args": browser_args,
            }
            
            if proxy:
                launch_opts["proxy"] = {"server": proxy}
            
            browser = await p.chromium.launch(**launch_opts)
            
            # Context com fingerprint randomizado
            viewport = get_random_viewport()
            context = await browser.new_context(
                user_agent=get_random_user_agent(),
                viewport=viewport,
                locale=random.choice(LOCALES),
                timezone_id=get_random_timezone(),
                color_scheme="light",
                device_scale_factor=random.choice([1, 1.25, 1.5, 2]),
            )
            
            # Headers extra
            await context.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            })
            
            page = await context.new_page()
            
            # Aplicar patches stealth
            await apply_stealth_patches(page)
            
            # Anexar proxy à página para uso em retry/reporting
            page.proxy_used = proxy
            
            logger.debug(f"Stealth browser criado: {viewport['width']}x{viewport['height']}")
            
            yield browser, context, page
            
    finally:
        if browser:
            await browser.close()


async def navigate_with_retry(
    page: Page, 
    url: str, 
    max_retries: int = 3,
    timeout: int = 30000
) -> bool:
    """
    Navega para URL com retry e detecção de bloqueio.
    
    Args:
        page: Página do Playwright
        url: URL destino
        max_retries: Número máximo de tentativas
        timeout: Timeout em ms
        
    Returns:
        True se sucesso, False se bloqueado/falhou
    """
    # Importar ProxyManager dinamicamente
    try:
        from infrastructure.proxy_manager import get_proxy_manager
        pm = get_proxy_manager()
    except ImportError:
        pm = None

    proxy_url = getattr(page, "proxy_used", None)

    for attempt in range(max_retries):
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            
            if response is None:
                logger.warning(f"Resposta None para {url} (tentativa {attempt + 1})")
                continue
            
            status = response.status
            
            # Detectar bloqueio
            if status == 403 or status == 429:
                logger.warning(f"⚠️ Bloqueio detectado ({status}) para {url}")
                
                # Reportar proxy queimado/falho
                if pm and proxy_url:
                    reason = f"Status {status} on {url}"
                    pm.report_dead_proxy(proxy_url, reason)
                    logger.info(f"🔥 Proxy reportado como falho: {proxy_url}")
                
                waitTime = 30 if status == 429 else 5
                await human_delay(waitTime, waitTime * 1.5)
                continue
            
            if status >= 400:
                logger.warning(f"⚠️ Erro {status} para {url}")
                continue
            
            # Verificar se é página de captcha
            content = await page.content()
            if "captcha" in content.lower() or "challenge" in content.lower():
                logger.warning(f"⚠️ Captcha detectado para {url}")
                
                # Reportar proxy como suspeito/falho
                if pm and proxy_url:
                    pm.report_dead_proxy(proxy_url, f"Captcha on {url}")
                
                return False
            
            # Sucesso! Se estavamos usando proxy, marcar sucesso
            if pm and proxy_url:
                pm.mark_success(proxy_url)
                
            return True
            
        except Exception as e:
            logger.warning(f"Erro navegando para {url}: {e} (tentativa {attempt + 1})")
            await human_delay(2, 5)
    
    return False
            
        except Exception as e:
            logger.warning(f"Erro navegando para {url}: {e} (tentativa {attempt + 1})")
            await human_delay(2, 5)
    
    return False
