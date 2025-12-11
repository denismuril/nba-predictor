# Integração da Gestão de Banca no Web App

## Instruções para Integrar

Para adicionar a interface de gestão de banca no `nba_predictor_web.py`, siga os passos:

### 1. Adicionar Import

No topo do arquivo `nba_predictor_web.py`, adicione:

```python
# Após as outras imports
from betting.web_ui import render_bankroll_management
```

### 2. Modificar TAB 2 (Sugestões de Aposta)

Localizar a linha (aprox. 836):

```python
# --- TAB 2: SUGESTÕES DE APOSTA ---
with tab2:
    st.header("💰 Sugestões de Aposta Baseadas em EV")
    # ... código antigo ...
```

Substituir por:

```python
# --- TAB 2: GESTÃO DE BANCA PROFISSIONAL ---
with tab2:
    render_bankroll_management(daily_games, bankroll_input, kelly_fraction)
```

### 3. Resultado

A tab agora exibirá:

- ✅ Stakes calculados automaticamente com Kelly
- ✅ Alertas visuais de correlação
- ✅ Progress bars do Kelly
- ✅ Resumo da sessão com métricas
- ✅ Monitor de CLV com gráficos

---

## Alternativa: Integração Manual

Se preferir fazer a integração manualmente sem modificar o arquivo grande, você pode:

1. Rodar o web app normal
2. Usar o módulo `betting/web_ui.py` como referência
3. A funcionalidade de staking já está disponível via `odds_shopping.py`

O sistema funciona em ambos os casos!
