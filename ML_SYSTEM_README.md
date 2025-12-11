# 🎯 Sistema de ML - Guia Completo

## 📊 Status Atual

**META SUPERADA! ✅**
- **Accuracy Final: 76.89%** (Isotonic Calibration)
- Baseline: 74.55%
- **Melhoria: +2.34%** 🚀

---

## 🚀 Comandos Principais

### 1. Monitoramento Contínuo

```bash
# Atualizar métricas diárias
python scripts/monitoring_system.py --update-daily

# Atualizar métricas semanais
python scripts/monitoring_system.py --update-weekly

# Verificar alertas
python scripts/monitoring_system.py --check-alerts

# Gerar relatório completo
python scripts/monitoring_system.py --generate-report
```

**Arquivos Gerados:**
- `data/monitoring/metrics_history.json` - Histórico de métricas
- `data/monitoring/alerts.json` - Alertas ativos
- `data/monitoring/performance_report.json` - Relatório completo

---

### 2. Fase 3 - Feature Interactions

```bash
# Testar impacto das interactions
python scripts/create_feature_interactions.py --test-interactions
```

**Features criadas:**
- Produtos: `home_sos × away_sos`, `home_efg × away_efg`
- Diferenças: `home_sos - away_sos`, `home_streak - away_streak`
- Ratios: `home_win / away_win`
- Composites: `sos × win_rate × rest_advantage`

---

### 3. Diagnóstico e Calibração

```bash
# Diagnosticar times problemáticos
python scripts/diagnose_problem_teams.py

# Calibrar modelo
python scripts/calibrate_model.py --method all

# Error analysis
python scripts/error_analysis.py --save-report
```

---

## 📈 Resultados Obtidos

### Fase 1: Quick Wins
| Item | Status | Resultado |
|------|--------|-----------|
| Advanced Features | ✅ | SOS #2 e #3 em importance |
| Feature Selection | ✅ | 165 → 86 features (-47.9%) |
| Hyperparameter Opt | ✅ | Optuna com 50 trials |
| **Resultado** | ✅ | **73.37% accuracy** |

### Fase 2: Calibração
| Método | Accuracy | Melhoria |
|--------|----------|----------|
| Original | 76.28% | - |
| Platt Scaling | 76.53% | +0.25% |
| **Isotonic** | **76.89%** | **+0.61%** ✅ |

**Modelo salvo:** `data/models/ensemble_model_calibrated_isotonic.joblib`

### Error Analysis
| Segmento | Accuracy | Insights |
|----------|----------|----------|
| Favoritos | 82.16% | ✅ Excelente |
| Underdogs | 65.00% | ⚠️ Viés detectado |
| Spreads 10+ pts | 97.19% | ✅ Blowouts perfeitos |
| Spreads 0-3 pts | 53.16% | ❌ Jogos competitivos |

---

## 🔍 Insights Importantes

### Times Problemáticos
- **Denver Nuggets:** 39% accuracy (gap de 187 dias = off-season)
- **Cleveland Cavaliers:** 52% accuracy (melhorando com tempo)
- **Ambos:** Mais erros antigos que recentes ✅

### Features Mais Importantes
1. `home_sos_10` (10.94%) 🌟
2. `home_rolling_10_win` (9.89%)
3. `away_sos_10` (7.51%) 🌟
4. `home_win_streak` (5.32%) 🌟

---

## 📁 Arquivos Principais

### Modelos
```
data/models/
├── ensemble_model_calibrated_isotonic.joblib  # USAR ESTE! ⭐
├── ensemble_model_final.joblib                # Backup
├── feature_names_final.joblib                 # 86 features selecionadas
├── best_ensemble_params.joblib                # Hiperparâmetros otimizados
└── best_totals_params.joblib                  # Totals otimizados
```

### Relatórios
```
data/models/
├── phase1_final_results.json          # Resultados Fase 1
├── calibration_results.json           # Resultados calibração
├── error_analysis_report.json         # Análise de erros
├── team_diagnosis_report.json         # Denver/Cleveland
└── feature_importance_ranking.csv     # Ranking de features
```

### Monitoramento
```
data/monitoring/
├── metrics_history.json       # Histórico diário/semanal
├── alerts.json                # Alertas ativos
└── performance_report.json    # Relatório completo
```

---

## 🎯 Próximos Passos

### Opção 1: Deploy e Backtesting
- ✅ Modelo pronto (76.89%)
- Fazer backtesting com dados reais
- Implementar em produção

### Opção 2: Continuar Fase 3
```bash
# Testar feature interactions
python scripts/create_feature_interactions.py --test-interactions

# Se melhorar, retreinar
python scripts/train_all_models.py
```

### Opção 3: Monitoramento Produção
```bash
# Setup cron job para monitoramento diário
# Adicionar ao crontab:
0 2 * * * cd ~/nba-predictor && python scripts/monitoring_system.py --update-daily --check-alerts
```

---

## 💡 Comandos Úteis

### Retreinar Tudo
```bash
# Com modelo calibrado atual
python scripts/train_all_models.py

# COM feature interactions (se testado e aprovado)
python scripts/create_feature_interactions.py --test-interactions
# Se melhoria > 0.5%, integrar e retreinar
```

### Ver Performance Atual
```bash
# Accuracy por segmento
python scripts/error_analysis.py --save-report

# Métricas históricas
python scripts/monitoring_system.py --generate-report
```

### Diagnóstico
```bash
# Times específicos
python scripts/diagnose_problem_teams.py --teams "Denver Nuggets,Cleveland Cavaliers"

# Walk-forward validation
python scripts/walk_forward_validation.py
```

---

## 🏆 Conquistas

- ✅ **+2.34% vs baseline** (74.55% → 76.89%)
- ✅ SOS descoberta como feature GOLD
- ✅ Feature selection reduzindo 47.9%
- ✅ Calibração Isotonic funcionando perfeitamente
- ✅ Sistema de monitoramento implementado
- ✅ Diagnóstico profundo criado

---

## 📞 Suporte

**Modelo em Produção:** `ensemble_model_calibrated_isotonic.joblib`  
**Features:** 86 selecionadas (lista em `feature_names_final.joblib`)  
**Accuracy:** 76.89%  
**Brier Score:** 0.1410  
**Log Loss:** 0.4016

**Pronto para deploy!** 🚀
