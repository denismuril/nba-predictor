# Changelog

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
