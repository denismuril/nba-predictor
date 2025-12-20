
## v27.4 - Props Intelligence Activation (20 Dez 2025)

### 🎯 Ativação do Sistema "Props Intelligence"

Nova camada de inteligência para identificação de apostas EV+ em Player Props usando regras heurísticas "Sniper".

### ✅ Novos Recursos

| Componente | Descrição |
|------------|-----------|
| **Sniper Engine** | Regras heurísticas: OVER (Avg > Line+10%) e UNDER (Avg < Line-10%) |
| **PropsProcessor** | Features: `diff_to_avg`, `last_5_hit_rate`, `implied_prob` |
| **ProxyManager** | Lê proxies de `config/proxies.txt` com rotação automática |
| **Stealth Integration** | Proxies "queimados" são reportados automaticamente |

### 📂 Arquivos Modificados/Criados

```
config/
└── proxies.txt                  # NOVO: Lista de proxies (template)

infrastructure/
└── proxy_manager.py             # UPDATE: Carrega de config/proxies.txt

data/processing/
└── props_processor.py           # UPDATE: Sniper features (diff, hit_rate)

ml_pipeline/
└── player_props_engine.py       # UPDATE: Nova função analyze_props()

data/scrapers/
└── stealth_browser.py           # UPDATE: Integração com ProxyManager

run_props_analysis.py            # NOVO: Script de verificação
```

### 🔧 Regras Sniper (Heurísticas)

```python
# OVER: Se Média > Linha + 10% E Odd > 1.85
# UNDER: Se Média < Linha - 10% E Odd > 1.85
# Retorna apenas apostas com EV > 5%
```

### 🧪 Resultado de Teste

```text
🔥 RECOMENDAÇÃO: Aposte no UNDER Trae Young points
   Média: 24.2 | Linha: 32.5 (-25.5% de vantagem)
   Odds: 1.95 | EV: 22.4%
```

---

## v27.3 - Multi-Source Odds Scraper (19 Dez 2025)

### 🌐 Sistema Multi-Fonte de Odds

Novo sistema robusto de coleta de odds de **5 fontes gratuitas** com fallback inteligente, eliminando dependência de APIs pagas.

### ✅ Novos Componentes

| Componente | Descrição |
|------------|-----------|
| **MultiSourceOddsScraper** | Orquestrador com fallback e execução paralela |
| **OddsAgoraScraper** | Scraper para oddsagora.com.br |
| **OddsScannerScraper** | Scraper para oddsscanner.com |
| **SportyTraderScraper** | Scraper para sportytrader.com |
| **OddsSharkScraper** | Scraper para oddsshark.com |
| **BaseSiteScraper** | Classe base abstrata com utilitários |

### 📂 Novos Arquivos

```
data/scrapers/odds_sites/
├── __init__.py
├── base_scraper.py      # Classe base + PlaywrightMixin
├── odds_agora.py        # OddsAgora (BR)
├── odds_scanner.py      # OddsScanner (BR)
├── sporty_trader.py     # SportyTrader (PT-BR)
└── odds_shark.py        # OddsShark (US)

data/scrapers/
└── multi_odds_scraper.py  # Orquestrador multi-fonte
```

### 🔧 Arquivos Modificados

- `data/scrapers/odds_web_scraper.py`:
  - Novo método `_extract_from_nuxt()` para extrair odds do `window.__NUXT__`
  - Tempo de espera Cloudflare aumentado para 20s
  - Novos seletores CSS para `.odd-box__value`

- `data/scrapers/odds_scraper.py`:
  - Adicionado TIER 0 (MultiSourceOddsScraper) na hierarquia de fallback

### 🎯 Hierarquia de Fallback

```
TIER 0: MultiSource (5 sites gratuitos)
    → OddsPedia → OddsAgora → OddsScanner → SportyTrader → OddsShark
TIER 1: OddsPedia (fallback direto)
TIER 2: TheOddsAPI (API paga)
```

### 🧪 Resultados de Teste

- **OddsAgora retornou 31 jogos** em teste real
- **8 testes pytest passaram** (TestOddsValidator)
- Todos os módulos integrados sem erros

---

## v27.0 - God Mode Architecture (End-to-End) (19 Dez 2025)

### ⚡ Arquitetura Autônoma para Player Props

Introdução do motor "God Mode" (End-to-End) focado na rentabilidade de Player Props com infraestrutura de resiliência e auditoria.

### ✅ 4 Pilares Implementados

| Pilar | Componente | Função |
|-------|------------|--------|
| **1. ETL Pipeline** | `data/processing/props_processor.py` | Integração de scrape, normalização e cálculo de features (L5, Rest, H2H). |
| **2. Proxy Manager** | `infrastructure/proxy_manager.py` | Rotação automática de IPs e proteção de identidade para evitar 403/429. |
| **3. EV+ Engine** | `ml_pipeline/player_props_engine.py` | Motor de inferência focado em Valor Esperado (EV) e classificação Sniper. |
| **4. CLV Tracker** | `analysis/clv_tracker.py` | Auditoria de qualidade das apostas baseada na linha de fechamento. |

### 🔍 Destaques Técnicos

- **Zero Mock Data:** Pipeline recusa predições se dados históricos reais estiverem ausentes.
- **Sniper Bets:** Classificação automática baseada em EV > 5% e Confiança > 60%.
- **Resiliência:** Browser Stealth agora consome proxies rotativos automaticamente via Singleton.
- **Auditoria:** `clv_tracker.py` registra o momento exato da aposta para comparação futura.

### 📂 Arquivos Modificados/Criados

```
data/processing/
└── props_processor.py     # NOVO: ETL Engine

infrastructure/
└── proxy_manager.py       # NOVO: Gestão de Proxies

analysis/
└── clv_tracker.py         # NOVO: Auditoria de Performance

ml_pipeline/
└── player_props_engine.py # UPDATE: Lógica EV+ e Sniper
```

---

## v27.2 - Systemd Automation & Stabilization (18 Dez 2025)

### ⚙️ Automação via Systemd

Substituição completa do `cron` por **Systemd Timers** para maior robustez e monitoramento.

- **Service & Timer**: `nba-predictor.service` e `nba-predictor.timer` configurados para rodar diariamente às 10:00 AM.
- **Wrapper Script**: `run_pipeline.sh` garante ambiente virtual ativado e execução limpa.
- **Logs Centralizados**: `logs/pipeline.log` e `logs/pipeline_error.log` integrados ao journald.

### 🐛 Bug Fixes Críticos

- **Action Network Scraper**: Corrigido parsing da API que retornava props com apenas um lado (single-sided). Scraper agora recupera ~90 props/dia (antes 0).
- **Player Name Normalization**: Adicionado suporte para nomes abreviados (`D. Daniels` → `Dyson Daniels`) e correção de filtro de times.
- **Odds Validator**: Refinada validação de vigorish (Overround) para rejeitar mercados com margem > 25% ou negativa (erro de arb).

### 🧪 Testes

- **Suite de Testes**: `pytest` rodando 100% (19 passed).
- **Integração**: Verificado fluxo de `nba_predictor_web.py` consumindo novos props.

# Changelog

---

## v27.1 - Player Props Integration (18 Dez 2025)

### 🎯 Player Props via Action Network API

Implementação completa de scraping de Player Props usando chamadas HTTP diretas à API do Action Network.

### ✅ Novos Componentes

| Componente | Descrição |
|------------|-----------|
| **ActionNetworkScraper** | Scraper HTTP direto (sem Playwright) |
| **PlayerProp dataclass** | Estrutura para props (player, line, odds) |
| **player_name_normalizer** | Fuzzy matching contra roster oficial |
| **OddsDataManager.fetch_player_props()** | Integração no manager de odds |

### 📂 Novos Arquivos

```
data/scrapers/
├── action_network_scraper.py  # Scraper de props via API
└── player_name_normalizer.py  # Normalização de nomes
```

### 🎯 Resultados

- **91 props extraídos** em teste real
- **API funcional**: `/projections/available` retorna 200 OK
- **Prop types**: Points, Rebounds, Assists, Steals, Blocks
- **Exemplo**: `Mikal Bridges: points 16.5 (O:1.90 U:1.97)`

### 🔧 Decisões de Design

- **Requests vs Playwright**: Optou-se por HTTP direto após 12+ tentativas com Playwright. API direta é 10x mais rápida e 100% confiável.
- **Players como lista**: API retorna `players` como array, não dict. Parser adaptado para ambos formatos.
- **Single-sided props**: Aceita props apenas com over ou under quando par não disponível.

---

## v27.0 - Enterprise Rest Advantage Features (17 Dez 2025)

### 🎯 Granular Rest & Travel Fatigue Features

Implementação de features avançadas de descanso e fadiga de viagem para capturar nuances além de simples detecção de Back-to-Back.

### ✅ Novas Features em `ml_pipeline/feature_engineering_v2.py`

| Feature | Descrição |
|---------|-----------|
| `net_rest_days` | Alias formal para `rest_advantage` (home - away) |
| `rest_disadvantage_home` | Flag: home em B2B enquanto away descansou 2+ dias |
| `rest_disadvantage_away` | Flag: away em B2B enquanto home descansou 2+ dias |

### ✅ Novas Features em `ml_pipeline/advanced_features.py`

| Feature | Descrição |
|---------|-----------|
| `home_travel_km_3d` | Km acumulados pelo home nos últimos 3 jogos |
| `away_travel_km_3d` | Km acumulados pelo away nos últimos 3 jogos |
| `travel_km_advantage` | `away_travel - home_travel` (positivo = home vantagem) |

**Anti-Leakage:** Implementado com `shift(1)` para evitar vazamento de dados do jogo atual.

### ✅ Novo Método em `ml_pipeline/elo_system.py`

Adicionado `prever_jogo_enterprise()` com penalidades granulares:

| Parâmetro | Valor | Impacto |
|-----------|-------|---------|
| `REST_PENALTY_PER_DAY` | 15 Elo | Por dia de diferença |
| `MAX_REST_PENALTY` | 60 Elo | Cap (~1.8 pts spread) |
| `TRAVEL_PENALTY_PER_1000KM` | 6 Elo | Por 1000km viajados |
| `MAX_TRAVEL_PENALTY` | 40 Elo | Cap para extremos |
| Fresh vs Exhausted bonus | ±20 Elo | Se adversário B2B e você descansado |

### 🔧 Integração no Pipeline

- `data_preparation.py` - Adicionadas features ao `SAFE_EXACT_COLS` whitelist
- `advanced_features.py` - `add_travel_km_last_3_days()` integrado em `add_domain_expert_features()`

---

## v26.2 - Critical Security & Logic Fixes (17 Dez 2025)

### 🔒 Correções de Segurança e Lógica Crítica

Atualização de emergência para garantir a integridade do sistema, focada em 3 pilares: calibração Elo, estanqueidade de dados (anti-leakage) e estabilidade de coleta de odds.

### ✅ Mudanças em `ml_pipeline/elo_system.py`

- **HCA Corrigido:** Constante `HCA_ELO` ajustada para **70** (antes 100), refletindo vantagem de casa real de ~2.1 pontos.
- **Fadiga B2B:** Implementada penalidade de **50 pontos Elo** (~1.5 pts) na função `calcular_vitoria_esperada` para times em Back-to-Back.

### 🛡️ Allowlist Rígida em `ml_pipeline/train_ensemble_v6.py`

- **Fim da Blacklist:** Removida a lista de exclusão `base_drop_cols` que era vulnerável a novas colunas.
- **Allowlist Imutável:** Implementada filtragem positiva estrita. Apenas colunas com prefixos seguros são permitidas:

  ```python
  SAFE_PREFIXES = ['rolling_', 'elo_', 'rest_', 'is_b2b', 'feat_', 'encoded_']
  ```

### 🔧 Estabilidade em `data/scrapers/odds_scraper.py`

- **Async Lock Fix:** Função `acquire_rate_limit_sync` reescrita para utilizar `asyncio.get_running_loop()`.
- **Prevenção de Deadlock:** Estratégia "Fail Open" implementada para evitar que a verificação de rate limit trave loops de eventos existentes.

---

## v26.1 - Elo System Calibration & Anti-Leakage Reinforcement (17 Dez 2025)

### 🎯 Calibração do Sistema Elo (NBA Moderna)

Ajustes finos nos parâmetros Elo conforme auditoria técnica para refletir as tendências da NBA moderna.

### ✅ Mudanças em `ml_pipeline/elo_system.py`

| Parâmetro | Antes | Depois | Impacto |
|-----------|-------|--------|---------|
| `HCA_ELO` | 100 (~3.0 pts) | **70** (~2.1 pts) | Corrige superestimação de favoritos em casa |
| `B2B_PENALTY` | N/A | **50** (~1.5 pts) | Penaliza times em Back-to-Back |

### 🔧 Novos Métodos/Parâmetros

- `calcular_vitoria_esperada(is_b2b=False)` - Aceita flag de Back-to-Back
- `prever_jogo(home_is_b2b, away_is_b2b)` - Retorna spread ajustado por fadiga
- Normalização de probabilidades quando ambos times têm ajustes B2B

### ⚠️ DEPRECATION WARNING em `config/constants.py`

Adicionado aviso de depreciação acima de `ALL_STARS_2025` sugerindo migração para métricas dinâmicas:

- USG% (Usage Rate) > 25%
- PER (Player Efficiency Rating) > 20
- Minutos jogados > 28 MPG

### 🛡️ Reforço Anti-Leakage em `ml_pipeline/train_ensemble_v6.py`

Implementada **segunda camada de proteção** com allowlist por prefixo explícito:

```python
SAFE_PREFIXES = ('feat_', 'roll_', 'rolling_', 'elo_', 'rest_', 
                 'interaction_', 'referee_', 'h2h_')
```

**Benefícios:**

- Filtragem dupla (1ª em `data_preparation.py`, 2ª em `train_ensemble_v6.py`)
- Colunas não permitidas são descartadas silenciosamente com log de warning
- Validação final contra features perigosas com `raise ValueError` se vazamento detectado

### 📊 Impacto Esperado na Precisão

| Ajuste | Impacto Esperado |
|--------|------------------|
| HCA reduzido | Spreads mais precisos em jogos em casa |
| Penalidade B2B | Captura fadiga de times em sequência |
| Allowlist reforçada | Elimina risco de novas colunas vazarem dados do futuro |

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
