# Changelog

---

## v21.12 - Health Check & System Integrity (13 Dez 2025)

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
