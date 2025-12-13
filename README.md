# 🏀 NBA Predictor v21.11 - Scraping First

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()
[![ML](https://img.shields.io/badge/ML-Ensemble%20V6-purple.svg)]()
[![Bot](https://img.shields.io/badge/Telegram-Sniper%20Bot-blue.svg)]()
[![PBP](https://img.shields.io/badge/PBP-Clean%20Metrics-red.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)
[![Season](https://img.shields.io/badge/Season-2025--26-orange.svg)]()

Sistema profissional de análise quantitativa e previsão de jogos da NBA usando **Advanced Machine Learning (Ensemble V6)**, **NLP Sentiment Analysis**, **PBP Clean Metrics (No Garbage Time)**, **Bot Sniper com Alertas EV**, **SafeKelly Bankroll Management** e **Multi-source data validation**.

---

## 🎉 Destaques v21.12 (13 Dez 2025) - **Health Check Enhanced**

### 🏥 Monitoramento de Integridade Completo

- ✅ **Full System Scan:** O `health_check.py` agora cobre 100% das novas integrações (Playwright, PBPStats, Odds Scraper).
- ✅ **Integration Validation:** Validação automática de dependências críticas antes da execução em produção.
- ✅ **Linting & Standards:** Código de monitoramento padronizado e livre de erros de estilo.
- ✅ **Stability:** Garantia de que todos os subsistemas essenciais estão operacionais antes do início do pipeline.

---

## 🎉 Destaques v21.11 (13 Dez 2025) - **Scraping First Strategy**

### 🕷️ Odds Gratuits com Scraper Web

- ✅ **Zero Cost:** Prioridade para scraper do OddsPedia (Playwright) antes de usar APIs pagas.
- ✅ **Smart Fallback:** Sistema tenta scraping -> falha silenciosa -> API de backup. TIER 1 = Gratuito.
- ✅ **Anti-Bot Tech:** Simulação humana avançada (Scroll, Delays, FakeAgent) para evitar bloqueios.
- ✅ **Data Integrity:** Validação rigorosa de nomes de times evita "alucinações" do scraper.

---

## 🎉 Destaques v21.10 (13 Dez 2025) - **PBPStats Clean Metrics**

### 📊 Métricas Livres de Garbage Time

- ✅ **Clean Data:** Integração com `pbpstats` filtra minutos irrelevantes de jogos decididos.
- ✅ **New Features:** `clean_off_rtg`, `clean_def_rtg` e `clean_pace` permitem modelagem mais precisa da força real dos times.
- ✅ **Robustez:** Fallback agressivo garante operação contínua mesmo se API falhar.

---

## 🎉 Destaques v21.9 (13 Dez 2025) - **Web Performance Duplicate Fix**

### 🐛 Correção de UI e Estabilidade

- ✅ **Duplicate Rows Fixed:** Resolvido problema de duplicação de linhas na aba de Performance do web app.
- ✅ **Optimized Merge:** Lógica de junção de dados otimizada para garantir unicidade de chaves.
- ✅ **Documentation Update:** Changelog e README atualizados para refletir correções.

---

## 🎉 Destaques v21.8 (12 Dez 2025) - **Injury System V2.2 & Clean Tests**

### 🏥 Sistema de Lesões Completamente Dinâmico (V2.2)

- ✅ **Dynamic Player Stats:** Adeus dados hardcoded! Importância de jogadores agora carregada dinamicamente via CSV (RAPM/Stats).
- ✅ **Regex Nominal:** Algoritmo aprimorado de normalização de nomes (lida com "Jr", sufixos e acentos).
- ✅ **API Resilience:** Testes de scraper ajustados para validação real de fallback (Selenium/Excel).
- ✅ **Legacy Compatibility:** Camada de compatibilidade transparente para módulos antigos.

### 🧪 Qualidade de Código & Testes

- ✅ **Zero Warnings:** Suite de testes `pytest` limpa de alertas de NaNs e suppressão de logs esperados.
- ✅ **Health Check 2.0:** Novo módulo de verificação profunda do subsistema de lesões.
- ✅ **Coverage:** Testes unitários expandidos para `injury_scraper_v2.py`.

---

## 🎉 Destaques v21.7 (12 Dez 2025) - **Spread Model Fixed**

### 🐛 Correção no Modelo de Spread (XGBoost)

- ✅ **XGBoost Type Error:** Corrigido erro crítico que impedia o treinamento do modelo de Spread.
- ✅ **Referees Cleanup:** Coluna `referees` removida do input do modelo (causa do erro de tipo).
- ✅ **Training Stability:** Pipeline `train_all_models` validado e executando 100% com sucesso.
- ✅ **Validation Results:** Spread Model atingiu **MAE de 2.51 pontos** após a correção.

---

## 🎉 Destaques v21.6 (12 Dez 2025) - **Telegram Bot Fix**

### 🐛 Correção Crítica no Bot do Telegram

- ✅ **Telegram Bot Fix:** Corrigido erro `market_line` que impedia a geração de resultados.
- ✅ **Odds Shopping:** Ajuste de compatibilidade para exibir corretamente Spreads e Moneyline.
- ✅ **Estabilidade:** Reinicialização de serviço para garantir aplicação da correção.

---

## 🎉 Destaques v21.9 (12 Dez 2025) - **Missing Games Fixed**

### 🚑 Correção de Resultados Faltantes ("None")

- ✅ **Full Date Alignment:** Script `align_prediction_dates` corrigiu +2900 previsões com data/fuso diferente da API.
- ✅ **Preseason Support:** `force_update_all_results.py` agora busca dados de Preseason e Regular Season.
- ✅ **Smart Insert:** Script de atualização agora faz INSERT de jogos novos (antes apenas UPDATE), recuperando ~800 jogos perdidos.
- ✅ **Team Normalization:** Suporte aprimorado para nomes completos ("Milwaukee Bucks" -> "MIL").

---

## 🎉 Destaques v21.6 (11 Dez 2025) - **Injury Cache System**

### 🏥 Sistema de Cache de Lesões

- ✅ **Cache-First Strategy:** Reduz scraping e risco de detecção de bot
- ✅ **TTL Configurável:** 30 minutos (via `INJURY_CACHE_TTL_MINUTES`)
- ✅ **Strategy Pattern:** Scrapers modulares (PDF, Rotowire, ESPN)
- ✅ **Thread-Safe:** Lock para operações de cache concorrentes

### 🔔 Alertas Automáticos de Lesões (Telegram)

- ✅ **Bot v20.8:** Novo job de alertas a cada 30 minutos
- ✅ **Jogadores de Alto Impacto:** Notifica quando MVP/All-Stars ficam OUT
- ✅ **Anti-Spam:** Cooldown de 30 min por alerta

### 📊 Arquivos Criados/Modificados

| Arquivo | Mudança |
|:---|:---|
| `injury_scraper_v2.py` | Refatorado com CacheManager |
| `injury_telegram_alerts.py` | Novo módulo de alertas |
| `nba_tigrinho_bot.py` | Atualizado para v20.8 |

---

## 🎉 Histórico de Versões Anteriores

<details>
<summary><b>v21.3 - Prediction Stability Fix</b></summary>

### 🐛 Bug Crítico Corrigido

- ✅ **100% Probability Bug Fixed:** Eliminado bug onde jogos retornavam 100%/0%.
- ✅ **Causa Raiz:** Features RAPM/BPM/Referee com inconsistência treino/predição.
- ✅ **Solução:** Features zeradas antes da predição para consistência.

</details>

## 🎉 Histórico de Versões Anteriores

<details>
<summary><b>v21.2 - Security Audit Fixes</b></summary>

### 🔒 Correções Críticas de Segurança

- ✅ **Data Leakage Eliminado:** BLACKLIST explícita bloqueia features de Smart Money.
- ✅ **Odds Estimadas Sinalizadas:** Sistema diferencia Fair Odds de Market Odds.
- ✅ **Spread Logístico Calibrado:** Fórmula inversa logística implementada.
- ✅ **SafeKellyStrategy:** Proteções contra ruína implementadas.

</details>

### 🔬 Auditoria Forense

- ✅ **EWMA Consistency:** Rolling features agora usam EWMA em todo o pipeline.
- ✅ **Smart Money Fix:** Features ignoradas quando sem dados reais de movimento de odds.
- ✅ **HCA Cap:** Aumentado de 4.5 → 5.0 pontos.
- ✅ **15 Módulos Auditados:** Análise completa de `ml_pipeline`, `core`, `market`, `betting`.

</details>

<details>
<summary><b>v21.0 - Data Leakage Fix & Infra Improvements</b></summary>

### 🔒 Eliminação de Data Leakage (Primeira Iteração)

- ✅ **Feature Whitelist:** `prepare_data_for_training` com whitelist restrita.
- ✅ **Prefixos Seguros:** Apenas `rolling_*`, `elo_*`, `rest_*`, `context_*`.
- ✅ **Bloqueio de Stats Raw:** Estatísticas do jogo atual bloqueadas.
- ✅ **dotenv Global:** `load_dotenv()` no escopo global.
- ✅ **HTTP 401/403:** Tratamento específico com mensagem clara.

</details>

<details>
<summary><b>v20.7 - Professional Bankroll Management</b></summary>

### 💰 Sistema de Gestão de Banca

- ✅ **Kelly Criterion Fracionado:** Cálculo automático de stakes usando Kelly/4 (0.25).
- ✅ **Hard Cap 3%:** Proteção absoluta contra overbet.
- ✅ **Detecção de Correlação:** Reduz stakes 50% para apostas relacionadas (mesmo jogo/time).
- ✅ **Monitor CLV:** Rastreamento de Closing Line Value para validação de mercado.
- ✅ **14 Testes Unitários:** 100% pass rate em gestão de risco.

</details>

<details>
<summary><b>v20.6 - Monte Carlo Dinâmico</b></summary>

- ✅ **HCA Adaptativo:** Simulação agora utiliza o **Home Court Advantage dinâmico** (Altitude, Torcida, Recency).
- ✅ **Precisão Aumentada:** Cenários extremos (ex: Denver em casa vs Lakers fora) agora refletem corretamente nas 1M simulações vetorizadas.
- ✅ **Validação Estatística:** Testes confirmam ajuste de probabilidade conforme força do mando de campo.

</details>

<details>
<summary><b>v20.5 - Auditoria de Segurança</b></summary>

- ✅ **Token Removido:** Telegram token agora **obrigatório via env var** (nunca hardcoded).
- ✅ **Fail-Fast:** Bot falha imediatamente se `TELEGRAM_BOT_TOKEN` não estiver configurado.
- ✅ **Schema Validation:** API de odds valida estrutura antes de processar (evita crashes).
- ✅ **Circuit Breaker:** Após 3 falhas consecutivas na API de odds, bloqueia requests por 5 minutos.

</details>

### 🏆 Machine Learning V6

- ✅ **Walk-Forward Validation:** Acurácia Temporal **61.85%** ±5.31%.
- ✅ **Calibração Isotônica:** Acurácia Calibrada **65.38%**.
- ✅ **145 Features Seguras:** Seleção via whitelist anti-leakage.

### 🤖 Telegram Bot Sniper (v20.8)

| Comando | Descrição |
|---------|-----------|
| `/start` | Boas-vindas (Admin via .env) |
| `/jogos` | Predições do dia com EV |
| `/props` | Top 3 jogadores (PTS, REB, AST) |
| `/news` | Alertas de lesão (Woj/Shams) |
| `/status` | Saúde do sistema |
| `/heartbeat` | Status dos jobs background |
| 🚨 Auto | Alertas quando EV > 5% |
| 🏥 Auto | Alertas de lesões críticas (30 min)

---

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone <seu-repo>
cd nba-predictor

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
# Instalar dependências
pip install -r requirements.txt

# Instalar navegadores para o Scraper
playwright install chromium
```

### 2. Configuração

Crie um arquivo `.env` na raiz:

```bash
ODDS_API_KEY=sua_chave_aqui
TELEGRAM_BOT_TOKEN=seu_token_aqui
DB_TYPE=postgres  # ou sqlite
```

### 3. Treinar Modelos (Recomendado)

```bash
# Treinar TODOS os modelos (recomendado)
python ml_pipeline/train_all.py
```

### 4. Executar

```bash
# 🤖 Telegram Bot (Modo Produção)
python telegram_bot/nba_tigrinho_bot.py

# 🖥️ Streamlit Dashboard
streamlit run nba_predictor_web.py

# 💻 CLI com Machine Learning
python main.py --ml

# 🔄 Orchestrador (Pipeline Completo)
python orchestrator.py
```

---

## 💻 Uso Detalhado

### Comandos do Bot (Telegram)

- `/start` - Boas-vindas (primeiro user = Admin para alertas).
- `/jogos` - Lista os jogos do dia com predições, spread justo e confiança.
- `/props` - Mostra Top 3 jogadores em Pontos, Rebotes e Assistências.
- `/news` - Últimas notícias de lesões (Twitter/X).
- `/status` - Verifica drift do modelo e saúde do sistema.
- `/testes` - Roda suite de testes de integridade.

> **Nota:** O primeiro usuário a enviar `/start` é registrado como Admin e recebe alertas SNIPER.

### Automação (Cron Job)

- O sistema atualiza resultados e roda modelos automaticamente todos os dias às **10:00 AM**.

- Comando: `scripts/orchestrator.sh`

### CLI Avançado

```bash
# Backtest de estratégia de apostas
python -m ml_pipeline.backtest_betting --days 90 --initial-bankroll 1000

# Otimização de Hiperparâmetros
python -m ml_pipeline.optimize_ensemble
```

---

## 📁 Estrutura do Projeto

```
nba-predictor/
├── core/                      # Lógica de negócio (Power Rating, Monte Carlo)
├── data/                      # Dados, Scrapers e Repositórios
│   ├── scrapers/              # Coleta de dados (NBA API, ESPN, Odds)
│   └── models/                # Modelos treinados (.joblib)
├── ml_pipeline/               # Pipeline de Machine Learning
│   ├── train_ensemble_v6.py   # ⭐ Modelo Principal (V6)
│   ├── train_spread_real.py   # Modelo de Spread
│   ├── feature_engineering_v2.py
│   └── predict.py             # Inferência
├── telegram_bot/              # 🤖 Bot do Telegram
│   └── nba_tigrinho_bot.py    # Script principal do bot
├── interfaces/                # Outras interfaces (CLI)
├── utils/                     # Utilitários (Kelly, Logs, Validação)
├── nba_predictor_web.py       # Dashboard Streamlit
├── main.py                    # Entry point CLI
└── requirements.txt
```

---

## 🧠 Metodologia ML V6

### Ensemble Stacking Não-Linear

O modelo V6 utiliza uma arquitetura de **Stacking** avançada:

1. **Base Learners:**
    - `RandomForestClassifier` (Robustez)
    - `XGBClassifier` (Gradiente Boosting)
    - `LGBMClassifier` (Velocidade e Precisão)
    - `HistGradientBoostingClassifier` (Eficiência em grandes datasets)
    - `ExtraTreesClassifier` (Redução de variância)

2. **Meta-Learner (Refatorado v18.1):**
    - `LogisticRegression` (Solver: liblinear, Penalty: L1)
    - L1 regularização para lidar com colinearidade entre modelos base.
    - Função: Calibrar as probabilidades brutas para confiança realística.

3. **Features (v21.2):**
    - EWMA Rolling Stats (mais reativo que SMA)
    - NLP Sentiment (tweets de Woj/Shams)
    - Player Impact (RAPM/BPM) - Top 5 jogadores por time
    - Referee Stats - Viés de arbitragem
    - ⚠️ Smart Money Signal - **EXCLUÍDO** (eliminação de data leakage v21.2)

---

## 💰 Sistema de Gestão de Banca Profissional

### 🎯 Kelly Criterion com Proteção

O sistema agora inclui gestão automática de banca usando **Kelly Criterion fracionado**:

```python
from betting.staking_strategy import KellyCriterionStrategy

strategy = KellyCriterionStrategy(bankroll=1000.0)
result = strategy.calculate_optimal_stake(
    model_prob=0.555,  # Prob do modelo
    market_odds=1.95,  # Odds oferecidas
    confidence=0.80
)

print(f"Apostar: ${result['stake_amount']:.2f}")  # Ex: $10.50
```

### 🛡️ Proteções Implementadas (v21.2 SafeKellyStrategy)

- ✅ **Kelly Fracionado (0.25)**: Reduz volatilidade em 4x
- ✅ **Hard Cap Dinâmico**: Reduz proporcional ao drawdown (base 3%)
- ✅ **Stop-Loss Diário (10%)**: Proteção contra ruína
- ✅ **Ajuste por Perdas**: Reduz Kelly 20% a cada 3 perdas consecutivas
- ✅ **Min Edge (5%)**: Filtro de apostas marginais
- ✅ **Validação de Odds**: Rejeita odds estimadas/fictícias

### 📊 Monitor CLV (Closing Line Value)

Valide a qualidade das suas apostas:

```bash
# Analisar performance dos últimos 30 dias
python scripts/monitor_clv.py --days 30
```

**Documentação completa**: [`betting/README.md`](betting/README.md)

---

## ⚠️ Avisos e Boas Práticas

### Gestão de Banca

- **NUNCA** aposte dinheiro que não pode perder.
- Use **Quarter Kelly** (0.25x) implementado no sistema.
- Sempre monitore seu CLV para validar edge sobre o mercado.
- O sistema é uma ferramenta de apoio à decisão, não uma garantia de lucro.

### Disclaimer

Este software é para fins educacionais e de pesquisa. O autor não se responsabiliza por perdas financeiras decorrentes do uso das predições.

---

## 📞 Suporte

**Versão:** v21.12 (Health Check Enhanced)  
**Status:** ✅ Production Ready  
**Temporada:** 2025-26  
**Última Atualização:** 13 Dezembro 2025  
**Novidades:** Health Check abrangente de novas integrações de scraping e stats.

**Documentação Adicional:**

- [`betting/README.md`](betting/README.md) - Guia completo de gestão de banca
- [`CHANGELOG.md`](CHANGELOG.md) - Histórico completo de alterações
