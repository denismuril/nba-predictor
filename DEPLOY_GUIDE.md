# 🚀 NBA Predictor - Guia de Deploy para Produção

**Data:** 2025-12-02  
**Versão:** 2.0 (com melhorias implementadas)  
**Status:** ✅ Pronto para deploy

---

## ✅ Pré-requisitos Verificados

- [x] Modelos retreinados com performance melhorada
- [x] Fórmulas canônicas implementadas
- [x] Kelly Criterion concurrent implementado
- [x] 32 testes unitários passando
- [x] Training-serving skew eliminado

---

## 📋 Checklist de Deploy

### 1. **Verificar Arquivos de Modelo** (2 min)

```bash
cd /home/denis/nba-predictor

# Verificar que os modelos foram atualizados hoje
ls -lh data/models/*.joblib | grep "Dec  2"

# Devem aparecer:
# - ensemble_model.joblib (Moneyline - 69.11%)
# - spread_model.joblib (Spread - 5.86 MAE)  
# - totals_model_v16.joblib (Totals - 15.31 MAE)
# - best_hyperparameters.joblib
```

✅ **Verificado:** Modelos atualizados às 15:41 e 17:45 de hoje

---

### 2. **Testar Predição End-to-End** (5 min)

```bash
# Teste rápido do pipeline completo
python -c "
from ml_pipeline.predict import predict_next_games
import pandas as pd

print('🧪 Testando pipeline de predição...')
try:
    results = predict_next_games()
    if results is not None and not results.empty:
        print(f'✅ Predições geradas: {len(results)} jogos')
        print(results[['home_team', 'away_team', 'prob_ml_home']].head())
        print('✅ Pipeline funcionando!')
    else:
        print('ℹ️  Sem jogos para hoje (normal se não houver jogos)')
except Exception as e:
    print(f'❌ Erro: {e}')
"
```

---

### 3. **Verificar Streamlit App** (3 min)

```bash
# Iniciar Streamlit (se ainda não estiver rodando)
streamlit run nba_predictor_web.py --server.port 8501
```

**O que verificar:**

- ✅ Dashboard carrega sem erros
- ✅ Predições aparecem (se houver jogos hoje)
- ✅ Métricas históricas carregam
- ✅ Player props funcionam

---

### 4. **Verificar Telegram Bot** (3 min)

```bash
# Testar bot (se estiver configurado)
python nba_tigrinho_bot.py
```

**Comandos para testar:**

- `/start` - Bot responde
- `/today` - Predições de hoje
- `/odds` - Comparação com odds
- `/props` - Player props

---

### 5. **Backup dos Modelos Antigos** (2 min)

```bash
# Criar backup dos modelos anteriores (se ainda existirem)
mkdir -p data/models/backup_pre_v2.0_$(date +%Y%m%d)

# Copiar modelos antigos se não fez backup ainda
cp data/models/ensemble_v7.joblib data/models/backup_pre_v2.0_$(date +%Y%m%d)/ 2>/dev/null || echo "Sem modelo V7 para backup"
```

---

### 6. **Configurar Monitoramento** (5 min)

#### A. Criar script de health check

```python
# scripts/health_check.py
"""
Health check script para monitoramento de produção.
"""
import sys
from pathlib import Path
from datetime import datetime

def check_models():
    """Verifica se modelos existem e são recentes."""
    models = ['ensemble_model.joblib', 'spread_model.joblib', 'totals_model_v16.joblib']
    
    for model in models:
        path = Path(f'data/models/{model}')
        if not path.exists():
            print(f'❌ Modelo ausente: {model}')
            return False
        
        # Verificar se modelo tem menos de 7 dias
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        
        if age_days > 7:
            print(f'⚠️  Modelo antigo ({age_days} dias): {model}')
    
    print('✅ Todos os modelos OK')
    return True

def check_pipeline():
    """Testa pipeline de predição."""
    try:
        from ml_pipeline.predict import load_current_model
        model, calibrator = load_current_model()
        print('✅ Pipeline de predição OK')
        return True
    except Exception as e:
        print(f'❌ Erro no pipeline: {e}')
        return False

if __name__ == "__main__":
    checks_passed = check_models() and check_pipeline()
    sys.exit(0 if checks_passed else 1)
```

#### B. Executar health check

```bash
python scripts/health_check.py
```

---

### 7. **Configurar Logging de Produção** (opcional)

```bash
# Criar diretório de logs se não existir
mkdir -p logs

# Adicionar ao crontab para logging diário
# crontab -e
# 0 9 * * * cd /home/denis/nba-predictor && python scripts/health_check.py >> logs/health_check.log 2>&1
```

---

## 🎯 Deploy Finalizado

### Performance Esperada

| Modelo | Métrica | Valor | vs Baseline |
|--------|---------|-------|-------------|
| **Moneyline** | Accuracy | 69.11% | +7.11pp ✅ |
| **Spread** | MAE | 5.86 pts | -30.7% ✅ |
| **Totals** | MAE | 15.31 pts | Estável ✅ |

### Melhorias de Risco

- ✅ Max Drawdown: ~42% → <20%
- ✅ Risk of Ruin: ~12% → ~3%
- ✅ Kelly Exposure: Limitado a 15% por noite

---

## 📊 Monitoramento Recomendado

### Métricas para Acompanhar Diariamente

1. **Accuracy Moneyline** (target: >68%)
2. **MAE Spread** (target: <6.5 pts)
3. **Total Exposure** (deve ser <15% em noites de múltiplos jogos)
4. **Feature Validation Warnings** (devem ser raros)

### Alertas para Configurar

```python
# Adicionar ao seu sistema de monitoramento
ALERTS = {
    'moneyline_accuracy': {'threshold': 0.65, 'window': '7d'},
    'spread_mae': {'threshold': 7.0, 'window': '7d'},
    'model_age': {'threshold': 7, 'unit': 'days'},
    'prediction_failures': {'threshold': 3, 'window': '1d'}
}
```

---

## 🔄 Rollback Plan (se necessário)

Se houver problemas graves:

```bash
# 1. Parar serviços
pkill -f streamlit
pkill -f nba_tigrinho_bot

# 2. Restaurar modelos antigos
cp data/models/backup_pre_v2.0_*/ensemble_v7.joblib data/models/ensemble_model.joblib

# 3. Reiniciar serviços
streamlit run nba_predictor_web.py &
python nba_tigrinho_bot.py &
```

---

## ✅ Deploy Checklist Final

- [ ] Modelos verificados (atualização de hoje)
- [ ] Teste de predição end-to-end passou
- [ ] Streamlit funcionando
- [ ] Telegram bot funcionando (se aplicável)
- [ ] Backup dos modelos antigos criado
- [ ] Health check executado com sucesso
- [ ] Monitoramento configurado

---

## 📞 Suporte

**Versão:** 2.0  
**Data de Deploy:** 2025-12-02  
**Melhorias Principais:**

- Kelly Criterion Concurrent
- Canonical NBA Formulas
- True Shooting % Features
- Grid Search Optimização (Spread)
- 32 Unit Tests

**Documentação Completa:** Ver `walkthrough.md`

---

🚀 **Sistema em Produção - Boa sorte com as apostas!** 🏀
