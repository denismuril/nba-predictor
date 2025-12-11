# Sistema de Gestão de Banca Profissional

## 📊 Visão Geral

Sistema completo de bankroll management para apostas esportivas usando **Kelly Criterion** com proteção contra ruína financeira.

---

## 🚀 Quick Start

### 1. Calcular Stake Ótimo

```python
from betting.staking_strategy import KellyCriterionStrategy

# Inicializar com sua banca atual
strategy = KellyCriterionStrategy(bankroll=1000.0)

# Calcular stake para uma aposta
result = strategy.calculate_optimal_stake(
    model_prob=0.555,  # 55.5% chance de ganhar
    market_odds=1.95,  # Odds oferecidas
    confidence=0.80    # Confidence do modelo
)

print(f"Apostar: ${result['stake_amount']:.2f}")
print(f"Edge: {result['edge']:.1f}%")
```

### 2. Usar com odds_shopping.py

```python
from market.odds_shopping import fetch_multi_bookie_odds, compare_lines

# Buscar odds do mercado
odds = fetch_multi_bookie_odds()

# Sua predição do modelo
prediction = {
    'Casa': 'LAL',
    'Prob Casa %': 55.5,
    'Spread Previsto': -3.5
}

# Comparar e calcular stakes automaticamente
opportunities = compare_lines(
    model_prediction=prediction,
    market_odds_data=odds,
    bankroll=1000.0,           # Banca atual
    calculate_stakes=True      # Calcular stakes
)

# Ver resultados
for opp in opportunities:
    if opp['recommendation'] == 'APOSTAR':
        print(f"\nTime: {opp['team']}")
        print(f"Mercado: {opp['market']}")
        print(f"Odds: {opp['odds']}")
        print(f"EV: {opp['ev']:.1f}%")
        print(f"Stake Sugerido: ${opp['suggested_stake_amount']:.2f}")
        
        if opp.get('correlation_alert'):
            print(f"⚠️ {opp['correlation_alert']}")
```

### 3. Monitorar CLV (Closing Line Value)

```bash
# Analisar últimos 30 dias
python scripts/monitor_clv.py --days 30

# Exportar relatório
python scripts/monitor_clv.py --days 30 --export reports/clv_report.csv
```

---

## 🛡️ Proteções de Risco

### Kelly Criterion Fracionado

- **Fração**: 0.25 (Kelly/4)
- **Objetivo**: Reduzir volatilidade mantendo crescimento ótimo
- **Resultado**: 4x menos variância que Kelly completo

### Hard Cap

- **Limite**: 3% da banca por aposta
- **Objetivo**: Proteção contra overbet mesmo com edges altos
- **Bypass**: Impossível exceder este limite

### Min Edge

- **Threshold**: 5% de Expected Value
- **Objetivo**: Filtrar apostas marginais
- **Resultado**: Apenas apostas com vantagem significativa

### Detecção de Correlação

- **Trigger**: Múltiplas apostas no mesmo jogo ou time
- **Ação**: Reduz stakes em 50% automaticamente
- **Exemplo**: LAL ML + LAL -3.5 → ambos stakes reduzidos

---

## 📈 Rastreamento de Apostas

### Estrutura do CSV

Arquivo: `data/bet_tracking.csv`

```csv
date,game_id,home_team,away_team,bet_team,market,odds_taken,odds_closing,stake_pct,stake_amount,model_prob,ev,result,profit,clv
2025-12-06,LAL_vs_BRK,LAL,BRK,LAL,Moneyline,1.95,1.89,1.05,10.50,55.5,8.5,WIN,9.98,3.17
```

### Adicionar Aposta Manualmente

```python
import pandas as pd
from pathlib import Path

bet_file = Path('data/bet_tracking.csv')
df = pd.read_csv(bet_file)

# Nova aposta
new_bet = {
    'date': '2025-12-06',
    'game_id': 'LAL_vs_BRK',
    'home_team': 'LAL',
    'away_team': 'BRK',
    'bet_team': 'LAL',
    'market': 'Moneyline',
    'odds_taken': 1.95,
    'odds_closing': None,  # Preencher após jogo
    'stake_pct': 1.05,
    'stake_amount': 10.50,
    'model_prob': 55.5,
    'ev': 8.5,
    'result': None,  # WIN/LOSS após jogo
    'profit': None,
    'clv': None
}

df = pd.concat([df, pd.DataFrame([new_bet])], ignore_index=True)
df.to_csv(bet_file, index=False)
```

---

## 📊 Interpretação de CLV

### CLV Positivo (+2% a +5%)

✅ **EXCELENTE** - Você está consistentemente conseguindo odds melhores que o mercado final.

**Ação**: Continuar estratégia atual

### CLV Neutro (-1% a +1%)

⚠️ **NORMAL** - Performance na média do mercado.

**Ação**: Buscar otimizações, considerar timing de entrada

### CLV Negativo (< -2%)

🚨 **ATENÇÃO** - Mercado fecha contra suas posições.

**Ação**: Revisar modelo ou estratégia de entrada

---

## 🧪 Testes

```bash
# Rodar testes unitários
pytest tests/test_staking_strategy.py -v

# Resultado esperado:
# ==================== 14 passed in 1.75s =====================
```

---

## ⚙️ Configuração Avançada

### Personalizar Parâmetros

```python
from betting.staking_strategy import KellyCriterionStrategy

strategy = KellyCriterionStrategy(
    bankroll=1000.0,
    kelly_fraction=0.25,    # Ajustar agressividade
    hard_cap_pct=0.03,      # Ajustar limite máximo
    min_edge_pct=0.05,      # Ajustar filtro de edge
    min_confidence=0.60     # Ajustar filtro de confidence
)
```

### Desabilitar Staking Automático

```python
# Calcular EV sem stakes
opportunities = compare_lines(
    model_prediction=prediction,
    market_odds_data=odds,
    calculate_stakes=False  # Desabilita staking
)
```

---

## 📖 Referências

### Kelly Criterion

- **Paper Original**: J.L. Kelly Jr. - "A New Interpretation of Information Rate" (1956)
- **Aplicação**: Maximiza crescimento logarítmico esperado da banca

### Fractional Kelly

- **Razão**: Kelly completo assume probabilidades perfeitas (irrealista)
- **Solução**: Usar fração (0.25 = Quarter Kelly) reduz risk of ruin

### CLV (Closing Line Value)

- **Definição**: Diferença entre odd apostada e odd de fechamento
- **Importância**: Melhor indicador de habilidade preditiva a longo prazo
- **Meta**: CLV médio positivo indica edge sobre o mercado

---

## 🤝 Suporte

Para dúvidas ou problemas:

1. Ver [`walkthrough.md`](file:///C:/Users/denis.santos/.gemini/antigravity/brain/b3a8dca2-ecdf-4279-b185-be55f7da7e58/walkthrough.md) - Documentação completa
2. Ver [`implementation_plan.md`](file:///C:/Users/denis.santos/.gemini/antigravity/brain/b3a8dca2-ecdf-4279-b185-be55f7da7e58/implementation_plan.md) - Detalhes técnicos
3. Executar demo: `python betting/staking_strategy.py`

---

## ⚠️ Disclaimer

Este sistema é para fins educacionais e de pesquisa. Apostas esportivas envolvem risco financeiro. Aposte com responsabilidade e apenas valores que pode perder.

**Gestão de banca adequada reduz risco, mas não elimina possibilidade de perdas.**
