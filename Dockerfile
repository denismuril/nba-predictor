# =============================================================================
# NBA Predictor - Multi-Stage Docker Build (Production Optimized)
# =============================================================================
# Features:
# - Multi-stage build para imagem final pequena (~500MB vs ~2GB)
# - Remove compiladores C++ após build (pandas, xgboost)
# - No hardcoded credentials
# - Non-root user for security
# - Health checks integrados
# =============================================================================

# ----- STAGE 1: Builder (compila dependências C++) -----
FROM python:3.12-slim AS builder

# Instalar compiladores necessários para pandas, numpy, xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libpq-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiar apenas requirements primeiro (layer caching)
COPY requirements.txt .

# Instalar dependências em diretório isolado
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Instalar Playwright e dependências do browser para scraping
RUN pip install --no-cache-dir --prefix=/install playwright \
    && /install/bin/playwright install chromium --with-deps

# ----- STAGE 2: Runtime (imagem final limpa) -----
FROM python:3.12-slim AS runtime

# Metadata
LABEL maintainer="NBA Predictor Team"
LABEL version="25.0"
LABEL description="NBA Predictor Go Live Edition - Production Image"

# Variáveis de ambiente (sem valores sensíveis!)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=America/Sao_Paulo

# Apenas runtime dependencies (sem compiladores)
# Inclui dependências do Playwright para execução de browser headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    tzdata \
    # Dependências do Playwright/Chromium
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Criar usuário non-root
RUN groupadd -r nba && useradd -r -g nba nba

# Copiar dependências instaladas do builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copiar código da aplicação
COPY --chown=nba:nba . .

# Criar diretórios necessários
RUN mkdir -p /app/data/cache /app/logs /app/models \
    && chown -R nba:nba /app

# Copiar e configurar entrypoint
COPY --chmod=755 entrypoint.sh /app/entrypoint.sh

# Switch para usuário non-root
USER nba

# Health check (verifica se Python + imports funcionam)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from infrastructure.database import AsyncDataManager; print('OK')" || exit 1

# Porta padrão Streamlit
EXPOSE 8501

# Entrypoint para configuração automática
ENTRYPOINT ["/app/entrypoint.sh"]

# Comando padrão (web interface)
# Opções: "web" (Streamlit), "bot" (Telegram), "cron" (scheduler)
CMD ["web"]

