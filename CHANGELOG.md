# Changelog

---

## v26.0 - Odds Module Refactoring (15 Dez 2025)

### 🔄 Arquitetura Multi-Provider para Odds

Refatoração completa do módulo de coleta de odds para priorizar **Web Scraping gratuito** sobre APIs pagas, com padrão Chain of Responsibility.

### ✅ Novos Componentes

| Componente | Descrição |
|------------|-----------|
| **OddsProvider Interface** | Interface abstrata para padronizar provedores |
| **SBRScraper** | Scraper do Sportsbook Review (TIER 1) |
| **OddsPediaProvider** | Adaptador para OddsPedia com JSON-LD (TIER 2) |
| **TheOddsAPIProvider** | API paga com controle de cota via Redis (TIER 3) |
| **OddsDataManager** | Orquestrador Chain of Responsibility |
| **QuotaExceededException** | Exceção para controle de cota da API |

### 📂 Novos Arquivos Criados

```
data/
├── interfaces/
│   └── odds_provider.py      # Interface OddsProvider + GameOdds dataclass
├── providers/
│   ├── sbr_scraper.py        # Scraper SBR (Playwright + stealth)
│   ├── oddspedia_provider.py # Adaptador OddsPedia
│   └── the_odds_api.py       # API paga com quota Redis
└── odds_manager.py           # OddsDataManager orchestrator

exceptions/
└── odds_exceptions.py        # QuotaExceededException
```

### 🔧 Arquivos Modificados

- `data/scrapers/odds_web_scraper.py` - Adicionado `_extract_from_json_ld()` para extração confiável via dados estruturados
- `Dockerfile` - Adicionado Playwright install em stages builder e runtime

### 🎯 Hierarquia de Fallback

```
SBRScraper (TIER 1, Gratuito)
    ↓ falha
OddsPediaProvider (TIER 2, Gratuito + JSON-LD)
    ↓ falha
TheOddsAPIProvider (TIER 3, Pago + Quota Control)
```

### 💰 Economia de Créditos

- **Controle de Cota:** Redis-based counter limita uso da API paga a 450/500 chamadas diárias
- **Prioridade Gratuita:** Scrapers gratuitos são tentados primeiro
- **JSON-LD Extraction:** Método mais confiável que seletores CSS

### 🐳 Docker Ready

```bash
# Dockerfile já inclui Playwright
docker-compose up -d
```

---

## v25.0 - Go Live Edition (14 Dez 2025)

### 🚀 Paper Trading & Shadow Mode

Sistema completo para validação sem risco antes de operar com dinheiro real.

### ✅ Novos Componentes

| Componente | Descrição |
|------------|-----------|
| **Paper Trading Engine** | Simula apostas com registro em PostgreSQL |
| **Settlement Script** | Liquida bets com resultados reais + PnL |
| **System Health Tab** | Novo tab no Streamlit com monitoramento |
| **STOP ALL BETS** | Botão de pânico para emergências |
| **Prefect Flows** | Orquestração profissional (substitui cron) |
| **Operations Runbook** | OPERATIONS.md com rotina diária |

### 📂 Novos Arquivos Criados

```
betting/
├── paper_trading.py     # Engine de paper trading
└── settle_paper_bets.py # Liquidação com PnL

flows/
├── daily_pipeline.py    # 4 Prefect flows
└── schedules.py         # Configuração de schedules

scripts/
└── setup_prefect.sh     # Setup automático

prefect.yaml             # Deploy config
OPERATIONS.md            # Runbook operacional
```

### 🔧 Arquivos Modificados

- `nba_predictor_web.py` - Novo tab "System Health" com:
  - Status PostgreSQL, Redis, Odds API
  - Paper Trading stats
  - Botão STOP ALL BETS
  
- `requirements.txt` - Adicionado:
  - `prefect>=2.14.0`
  - `prefect-shell>=0.2.0`

### ⏰ Flows Agendados (BRT)

| Horário | Flow | Descrição |
|---------|------|-----------|
| 08:00 | health-check | Verificação matinal |
| 09:00 | settlement | Liquidar paper bets |
| 17:00 | daily-pipeline | Previsões + Alertas |
| 18:00 | paper-trading | Capturar sinais |

### 🎯 Como Usar

```bash
# 1. Setup Prefect
bash scripts/setup_prefect.sh

# 2. Iniciar servidor + worker
prefect server start
prefect worker start --pool default-agent-pool

# 3. Acessar UIs
# Prefect: http://localhost:4200
# Streamlit: http://localhost:8501
```

---

## v23.1 - Correções de Imports e Validação (14 Dez 2025)

### 🔧 Correções Críticas

| Correção | Descrição |
|----------|-----------|
| **Import Errors** | Corrigidos imports obsoletos no orchestrator e ETL flow |
| **GameSchema Validation** | Corrigida geração de `game_id` e adicionados aliases de times |
| **APIs Removidas** | Removidas etapas de Twitter e Odds (keys expiradas) do pipeline |

### 📂 Arquivos Modificados

- `orchestrator.py`:
  - `get_todays_games` → `obter_schedule`
  - `get_all_injuries` → `InjuryManager().get_latest_injuries`
  - Removidas etapas Twitter Sentiment e Fetch Real Odds (APIs pagas)

- `etl/flows/daily_data_flow.py`:
  - `get_todays_games` → `obter_schedule`
  - `get_today_odds` → `obter_odds`
  - `get_all_injuries` → `InjuryManager().get_latest_injuries`
  - Corrigida geração de `game_id` (normaliza antes de gerar)

- `etl/schemas/__init__.py`:
  - Adicionados aliases de times: `BRK`, `CHO`, `PHO`
  - Migrados decorators Pydantic v1 → v2 (`@model_validator`, `@field_validator`)

- `feature_store/store.py`:
  - Corrigidas 4 queries SQL para PostgreSQL (GROUP BY com AVG)

### ✅ Verificação

- Health check: ✅ Passou
- Pytest: ✅ Todos os testes passaram
- Pipeline: ✅ 8 jogos, 19 lesões, 8 previsões geradas

---

## v23.0 - Grande Integração Enterprise (14 Dez 2025)

### 🚀 A Grande Integração

Fusão completa da infraestrutura Enterprise com o fluxo principal do sistema. Eliminação de scripts legados e subprocess.

### ✅ Novos Componentes

| Componente | Descrição |
|------------|-----------|
| **Enterprise Orchestrator** | Orquestrador async com Prefect nativo |
| **Sniper Engine** | Monitor de odds em tempo real (30s polling) |
| **Migration Script** | Migração v1→v2 com validação Pydantic |
| **Docker Enterprise** | Compose com PostgreSQL + Redis + Prefect + MLflow |

### 📂 Novos Arquivos Criados

```
betting/
└── sniper_engine.py       # Monitor de odds em tempo real

scripts/
└── migrate_v1_to_v2.py    # Script de migração de dados
```

### 🔧 Arquivos Modificados

- `orchestrator.py` - **Reescrito completamente** para v23.0:
  - AsyncDataManager em vez de db_manager legado
  - Prefect flows executados nativamente
  - asyncio.gather para paralelização
  - RedisCache integrado
  - Eliminação de todos os subprocess.run
  
- `docker-compose.yml` - Atualizado com novos serviços:
  - PostgreSQL 15 (persistência)
  - Redis 7 (cache distribuído)
  - Prefect Server (orquestração UI - porta 4200)
  - MLflow Server (tracking de modelos - porta 5000)
  - Sniper Engine (monitor de odds)
  - Telegram Bot (alertas)
  
- `scripts/health_check.py` - Novos checks:
  - Sniper Engine status
  - Prefect Server disponibilidade
  - Docker containers health

### 🎯 Sniper Engine - Detecção de Valor

Nova engine de apostas em tempo real:

```python
# Monitora Redis a cada 30 segundos
# Detecta quando: Minha Odd > Casa + 5%
# Integra Kelly Criterion para stake recomendado
# Alertas automáticos via Telegram
```

**Funcionalidades:**

- Line Movement Detection (mudanças bruscas)
- Fair Price via FeatureStore
- Kelly Criterion integrado
- Alertas Telegram automáticos
- Max 3 alertas por jogo (anti-spam)

### 🐳 Docker Compose Enterprise

```bash
# Subir toda a infraestrutura
docker-compose up -d

# Verificar status
docker-compose ps

# Serviços disponíveis:
# - Web: http://localhost:8501
# - Prefect: http://localhost:4200
# - MLflow: http://localhost:5000
```

### ⚠️ Tratamento de Erros (API de Odds)

```
Scraper OddsPedia --[timeout]--> Circuit Breaker --[OPEN]--> Fallback TheOddsAPI
                                                --[falha]--> Fair Odds Calculadas
```

---

## v22.0 - Enterprise Edition (14 Dez 2025)

### 🏢 Infraestrutura Enterprise-Grade

A maior atualização do sistema! Migração completa para arquitetura distribuída com PostgreSQL, Redis e padrões de resiliência.

### ✅ Novos Componentes

| Componente | Descrição |
|------------|-----------|
| **PostgreSQL** | Banco de dados principal (2820 jogos migrados) |
| **Redis 7.0** | Cache distribuído para lesões, bankroll e rate limiting |
| **AsyncDataManager** | Operações assíncronas com SQLAlchemy + asyncpg |
| **Circuit Breakers** | 3 registrados para resiliência de APIs |
| **Rate Limiters** | 6 APIs com controle de taxa via Redis |
| **Logs Estruturados** | Formato JSON para observabilidade |
| **Feature Store** | Point-in-time correctness para ML |
| **Dead Letter Queue** | Tratamento de falhas de validação |

### 📂 Novos Arquivos Criados

```
infrastructure/
├── __init__.py
├── database.py          # AsyncDataManager
├── redis_cache.py       # RedisCache
├── circuit_breaker.py   # CircuitBreaker
├── rate_limiter.py      # DistributedRateLimiter
└── logging_config.py    # Logs JSON

feature_store/
├── __init__.py
└── store.py             # FeatureStore

etl/
├── __init__.py
├── schemas/__init__.py  # Pydantic Schemas
├── dead_letter_queue.py # DLQ
└── flows/daily_data_flow.py
```

### 🔧 Arquivos Modificados

- `orchestrator.py` - Integrado logs estruturados e Circuit Breakers
- `scripts/health_check.py` - Adicionado `check_enterprise_infra()`
- `data/scrapers/odds_scraper.py` - Integrado Rate Limiter
- `requirements.txt` - Novas deps: asyncpg, aiosqlite, redis, pydantic, prefect
- `.env` - Configurações de PostgreSQL e Redis

### 🗑️ Limpeza Realizada

- Removidos bancos SQLite antigos (`nba_predictor.db`, `data/nba_games.db`)
- Removida pasta `deprecated/` e backup
- Removidos 38 modelos antigos de `data/models/backup_old/`
- Removidos arquivos `.bak` de modelos

---

## v21.13 - OddsPedia Scraper Stability Fix (13 Dez 2025)

### 🕷️ Correção de Scraper

- **Timeout Fix:** Aumentado timeout de navegação para 60s e alterada estratégia de espera para `domcontentloaded` para evitar erros de `Timeout 30000ms exceeded`.
- **Robustez:** Adicionado `wait_for_selector` explícito para garantir que o conteúdo dinâmico (jogos) foi carregado antes da extração.
- **Debug:** Implementado dump automático de HTML (`debug_oddspedia.html`) quando nenhum jogo é encontrado, facilitando diagnóstico de falhas de seletor.

---

### 🏥 Integridade de Sistema Aprimorada

- **New Health Checks:** Script `health_check.py` atualizado para validar novas integrações críticas:
  - **Playwright:** Verifica se a biblioteca de automação está instalada para o Odds Scraper.
  - **PBPStats:** Valida a presença do cliente de estatísticas limpas.
  - **Odds Web Scraper:** Garante que o módulo de scraping de odds esteja acessível.
- **Dependency Fix:** Identificada e resolvida dependência faltante (`joblib`) no ambiente de produção.
- **Linting & Quality:** Correção de estilo e formatação no script de verificação de saúde.

---

## v21.11 - Scraping First for Odds (13 Dez 2025)

### 🕷️ Estratégia "Scraping First"

- **New Scraper:** Implementado `OddsPediaScraper` usando **Playwright** + **BeautifulSoup** para capturar odds em tempo real sem custo.
- **Anti-Detection:** Rotacionamento de User-Agent, delays aleatórios e simulação de scroll humano para evitar bloqueios.
- **Tiered Fallback:** Nova hierarquia de coleta prioriza fontes gratuitas:
    1. **OddsPedia (Scraper)** - Gratuito
    2. **TheOddsAPI** - Pago/Limitado
    3. **SportsDataIO** - Pago/Limitado
    4. **RapidAPI** - Limitado
- **Cost Reduction:** Redução drástica no consumo de cotas de APIs pagas.

---

## v21.10 - PBPStats Integration & Clean Metrics (13 Dez 2025)

### 📊 Métricas Limpas (Sem Garbage Time)

- **Integration:** Adicionada biblioteca `pbpstats` para filtrar minutos irrelevantes de jogos decididos.
- **New Metrics:** Introduzidas `clean_off_rtg`, `clean_def_rtg` e `clean_pace` no pipeline de Feature Engineering.
- **Garbage Time Filter:** Filtra automaticamente possessões nos últimos 5 minutos com diferença > 15 pontos.
- **Robustez:** Implementado Fallback Agressivo no `pbp_client` para garantir que o pipeline nunca quebre se a API falhar.

---

## v21.9 - Web Performance Duplicate Fix (13 Dez 2025)

### 🐛 Correção de UI

- **Duplicate Rows:** Corrigido problema de duplicação na aba Performance do web app, causado por chaves de junção não-únicas.
- **Deduplication Logic:** Adicionada etapa explícita de `drop_duplicates` antes do merge de previsões com resultados históricos em `nba_predictor_web.py`.

---

## v21.8 - Injury System Reborn & Clean Tests (12 Dez 2025)

### 🏥 Sistema de Lesões v2.2

- **Refatoração Completa:** O script `injury_scraper_v2.py` foi reescrito para eliminar dados hardcoded.
- **Dynamic Stats:** Importância dos jogadores é calculada dinamicamente via `StatsManager` integrando dados RAPM e Estatísticas básicas.
- **Normalização Robusta:** Regex aprimorada para lidar com sufixos de nomes (Jr., II, III) garantindo casamento perfeito entre relatórios de lesão e banco de dados.
- **Cache v2:** Estrutura de cache simplificada e robusta, com validação de TTL e compatibilidade retroativa.

### 🧪 Testes e Qualidade

- **Zero Warnings:** Suite de testes `pytest` limpa.
  - Testes de `data_preparation` ajustados com maior volume de dados para evitar NaN warnings de rolling features.
  - Testes de `staking_strategy` agora suprimem logs de alerta esperados para uma saída limpa.
- **Health Check Enhanced:** `scripts/health_check.py` agora valida a integridade do `InjuryManager` e `StatsManager`.

---

## v21.7 - Spread Model Fixed (12 Dez 2025)

### 🐛 Correção Crítica de Treinamento

- **XGBoost Type Error Fix:** Resolvido erro que impedia o treinamento do modelo de Spread devido à coluna `referees` (object/string) ser passada para o XGBoost.
- **Feature Cleanup:** Coluna `referees` removida explicitamente do conjunto de features em `train_spread_real.py`.
- **Validação:** Pipeline `train_all_models` executado com sucesso completo.

---

## v21.6 - Telegram Bot Fix (12 Dez 2025)

### 🐛 Correção Crítica

- **Telegram Bot Error:** Resolvido erro `KeyError: 'market_line'` que impedia a geração de resultados `/jogos`.
- **Compatibilidade de Odds:** Ajustada lógica para tratar corretamente chaves `line` (Spreads) e `odds` (Moneyline) do novo formato do módulo `odds_shopping`.
- **Estabilidade:** Reinicialização forçada do serviço para garantir aplicação da correção em produção.

---

## v21.5 - Forensic Audit Complete (11 Dez 2025)

### 🔍 Auditoria Forense Completa

**Resumo:** Auditoria linha-por-linha realizada por Arquiteto de Sistemas Sênior. Todas as fórmulas matemáticas validadas. Proteções anti-leakage confirmadas robustas.

### ✅ Validações Concluídas

| Componente | Status |
|------------|--------|
| Elo Rating (elo_system.py) | ✅ Fórmula correta |
| Four Factors (data_preparation.py) | ✅ Fórmulas NBA.com |
| Monte Carlo (simulation.py) | ✅ Vetorizado, STD=12.5 |
| Kelly Criterion (staking_strategy.py) | ✅ Conservador (0.25) |
| Data Leakage Protection | ✅ Whitelist + FORCE_DROP |

### 🗑️ Limpeza de Código

- **25 arquivos movidos para `deprecated/`**
  - 10 de ml_pipeline/ (train_ensemble_v1-v5, feature_pipeline_v1-v3)
  - 12 de models/ (ensemble_v7*.joblib)
  - 3 da raiz (check_*.py, send_test.py)

### 🔧 Consolidação

- Criado `ml_pipeline/__init__.py` com aliases:
  - `train_model` → train_ensemble_v6
  - `prepare_features` → feature_pipeline_v4
- `.gitignore` atualizado para ignorar deprecated/

---

## v21.4 - Data Leakage Audit Complete (11 Dez 2025)

---

# Changelog v21.0 - Data Leakage Fix & Infra Improvements

**Data:** 08 Dezembro 2025

---

## 🔒 Correções Críticas de Data Leakage

### ml_pipeline/data_preparation.py

**Problema:** O modelo estava recebendo estatísticas "raw" do jogo atual (`home_efg`, `off_rating`) que são calculadas com o placar final. Isso é **Data Leakage** - o modelo via o resultado antes de tentar prever.

**Solução:** Mudança de **BLACKLIST** para **WHITELIST** na função `prepare_data_for_training`:

```python
# ANTES (Blacklist - PERIGOSO)
drop_cols = ['date', 'home_team', 'home_score', ...]
X = df.drop(columns=drop_cols)

# DEPOIS (Whitelist - SEGURO)
SAFE_PREFIXES = ('rolling_', 'elo_', 'rest_', 'context_', ...)
X = df[[c for c in df.columns if any(c.startswith(p) for p in SAFE_PREFIXES)]]
```

**Features Permitidas:**

- `rolling_*` - Médias móveis históricas
- `elo_*` - Elo ratings  
- `rest_*` - Dias de descanso
- `context_*` - Features contextuais
- `interaction_*` - Features de interação
- `referee_*` - Stats de árbitros
- `h2h_*` - Head-to-head histórico

### Smart Money Fix

**Problema:** O código passava `odds_home` duplicado como abertura e fechamento, gerando dados falsos de movimentação zero.

**Solução:** Verificação de existência de `opening_odds`/`closing_odds`. Se não existir, cria features `NaN` (desconhecido) em vez de zeros falsos.

---

## 🔧 Infraestrutura de Odds

### data/scrapers/odds_scraper.py

#### 1. dotenv Global

```python
from dotenv import load_dotenv

# CRITICAL: Carregar .env no escopo global
load_dotenv()
```

**Motivo:** Garantir que variáveis de ambiente sejam lidas independente de quem importou o módulo.

#### 2. Debug de API Keys

```python
def __init__(self, api_key=None):
    self.api_key = api_key or os.getenv('ODDS_API_KEY')
    logger.debug(f"TheOddsAPI: API Key loaded = {bool(self.api_key)}")
```

**Clientes atualizados:** TheOddsAPI, SportsDataIO, RapidAPI

#### 3. Tratamento HTTP 401/403

```python
except requests.exceptions.HTTPError as e:
    if e.response.status_code in (401, 403):
        logger.error("❌ TheOddsAPI: Chave API inválida ou expirada!")
    raise
```

---

## 📊 Modelo V6 Retreinado

| Métrica | Valor |
|---------|-------|
| Walk-Forward (5-Fold) | **61.85% ±5.31%** |
| Acurácia Calibrada | **65.38%** |
| Features Seguras | **145** |
| Jogos de Treino | 2,860 |

---

## Arquivos Modificados

- `ml_pipeline/data_preparation.py` - Whitelist de features + Smart Money fix
- `data/scrapers/odds_scraper.py` - dotenv global + HTTP 401/403 + debug logs

## Arquivos Criados

- `verify_leakage_fix.py` - Script de verificação das correções
