# 📋 OPERATIONS.md - Runbook de Operação Diária

## 🎯 NBA Predictor v24.0 - Go Live Operations

Este documento descreve a rotina operacional ideal para operar o sistema NBA Predictor em modo de produção.

---

## 🔄 Prefect Orchestration (Automático)

O sistema usa **Prefect** para orquestração profissional. Todos os jobs rodam automaticamente.

### Setup Inicial (Uma Vez)

```bash
# Instalar e configurar
bash scripts/setup_prefect.sh

# Terminal 1: Iniciar servidor
prefect server start

# Terminal 2: Iniciar worker
prefect worker start --pool default-agent-pool
```

### Flows Agendados (Automático)

| Horário (BRT) | Flow | Descrição |
|---------------|------|-----------|
| **08:00** | `health-check` | Verificação matinal do sistema |
| **09:00** | `settlement` | Liquidar paper bets de ontem |
| **17:00** | `daily-pipeline` | Fetch + Previsões + Alertas |
| **18:00** | `paper-trading` | Capturar sinais Sniper |

### UI de Monitoramento

```
http://localhost:4200
```

### Comandos Manuais

```bash
# Rodar pipeline manualmente
prefect deployment run 'NBA Daily Pipeline/daily-pipeline'

# Rodar settlement para data específica
python flows/daily_pipeline.py settle --date 2024-12-14

# Ver logs de um flow
prefect flow-run logs <flow-run-id>
```

## 🕐 Considerações de Fuso Horário

### NBA Schedule (Eastern Time → Brasília)

- **Jogos 19:00 ET** = 21:00 BRT
- **Jogos 22:00 ET** = 00:00 BRT
- **Jogos West Coast 22:30 ET** = 00:30 BRT

### Janela Ideal de Operação

```
17:00 BRT - Rodar previsões (2h antes dos primeiros jogos)
18:30 BRT - DEADLINE para apostas (30min antes do tip-off)
09:00 BRT+1 - Liquidar resultados do dia anterior
```

---

## 📊 Mode de Operação

### 1️⃣ Paper Trading (Primeiros 7 Dias)

```bash
# Iniciar simulação
python betting/paper_trading.py --bankroll 1000

# Verificar apostas registradas
python betting/paper_trading.py --report

# Liquidar dia anterior
python betting/settle_paper_bets.py --date YYYY-MM-DD
```

### 2️⃣ Shadow Mode (Semana 2-4)

- Apostas em valor mínimo (1% Kelly)
- Monitoramento intensivo
- Comparação com paper trading

### 3️⃣ Go Live (Após validação)

- Kelly fraction progressivo (10% → 25%)
- Stop-loss 10% diário
- Revisão semanal obrigatória

---

## 📱 Interpretando Alertas do Telegram

### ✅ Alerta Verde - Value Bet Detectada

```
🎯 VALUE BET DETECTADA
LAL @ BOS | Moneyline Home
Odds: 1.85 | Fair: 1.72 | Edge: 7.5%
Confiança: 82% | Kelly: 2.1%
```

**Ação**: Verificar no dashboard, considerar aposta

### 🟡 Alerta Amarelo - Movimento de Linha

```
📊 LINE MOVEMENT
PHX vs DEN | Home odds 1.80 → 1.95
Movimento: +8.3%
```

**Ação**: Investigar causa (lesão? sharp money?)

### 🔴 Alerta Vermelho - Trap Odds

```
🪤 TRAP DETECTADA
MIA vs NYK | 75% público em HOME
Linha movendo CONTRA público
```

**Ação**: EVITAR esta aposta

### 🏥 Alerta de Lesão

```
🏥 INJURY ALERT
LeBron James (LAL) - OUT
Impacto: Alto | Confiança ajustada: 0%
```

**Ação**: Bet automaticamente bloqueada

---

## 🚨 Procedimentos de Emergência

### Parar Todas as Apostas

1. Acesse: `streamlit run nba_predictor_web.py`
2. Tab: **System Health**
3. Clique: **🛑 STOP ALL BETS**

Ou via terminal:

```bash
touch data/.STOP_ALL_BETS
```

### Reativar Sistema

```bash
rm data/.STOP_ALL_BETS
```

### Redis Down

```bash
# Verificar
redis-cli ping

# Reiniciar
sudo systemctl restart redis

# Fallback: sistema opera sem cache
```

### PostgreSQL Down

```bash
# Verificar
pg_isready -h localhost -p 5432

# Reiniciar
sudo systemctl restart postgresql
```

---

## 🔄 Manutenção Semanal

### Segunda-feira (Manhã)

- [ ] Revisar performance da semana anterior
- [ ] Executar relatório: `python betting/paper_trading.py --report --days 7`
- [ ] Verificar calibração do modelo
- [ ] Limpar cache antigo

### Sexta-feira (Noite)

- [ ] Backup do banco de dados
- [ ] Verificar espaço em disco
- [ ] Atualizar dados históricos

---

## 📈 KPIs de Monitoramento

| Métrica | Target | Alerta |
|---------|--------|--------|
| Win Rate | > 52% | < 48% |
| ROI | > 5% | < 0% |
| CLV Médio | > 1% | < 0% |
| Edge Médio | > 3% | < 2% |
| Drawdown | < 15% | > 20% |

---

## 📞 Contatos

- **Telegram Alertas**: @nba_tigrinho_bot
- **Dashboard**: <http://localhost:8501>
- **Logs**: `logs/orchestrator.log`

---

## 📝 Changelog Operacional

| Data | Mudança | Responsável |
|------|---------|-------------|
| 2024-12-14 | Go Live v24.0 - Paper Trading | Sistema |
| - | - | - |
