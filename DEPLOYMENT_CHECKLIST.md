# 🚀 NBA Predictor - Deployment Checklist

## Pré-Deployment

### Ambiente
- [ ] Python 3.11+ instalado
- [ ] Todas as dependências instaladas (`pip install -r requirements.txt`)
- [ ] Database configurado (PostgreSQL ou SQLite)
- [ ] Permissões de escrita em `logs/`, `models/`, `monitoring/`, `reports/`

### Backups
- [ ] Backup do modelo atual (`models/ml_model.joblib`)
- [ ] Backup do calibrator (`models/calibrator.pkl`) se existir
- [ ] Backup do database
- [ ] Git commit com checkpoint

```bash
# Executar antes de deploy
git add .
git commit -m "Pre-deployment checkpoint - Sprint 1 & 2 complete"
git tag -a v1.0-production -m "Production deployment Sprint 1 & 2"
```

---

## Validação de Componentes

### Sprint 1
- [ ] Connection Pool testado
  ```bash
  python -c "from utils.connection_pool import ConnectionPool; print('✅ OK')"
  ```
- [ ] Team Normalization testado
  ```bash
  python -c "from utils.team_normalization import normalize_team; print(normalize_team('Lakers')); print('✅ OK')"
  ```
- [ ] Leakage Prevention testado
  ```bash
  python utils/leakage_prevention.py
  ```

### Sprint 2 P2.1
- [ ] AutoCalibrator testado
  ```bash
  python ml_pipeline/calibrator.py
  ```
- [ ] Calibration Monitor testado
  ```bash
  python monitoring/calibration_monitor.py
  ```

### Sprint 2 P2.2
- [ ] Domain Features testado
  ```bash
  python ml_pipeline/advanced_features.py
  ```
- [ ] Feature Pipeline testado
  ```bash
  python ml_pipeline/feature_pipeline.py
  ```

### Integration
- [ ] Smoke tests passando (5/5)
  ```bash
  python tests/smoke_tests.py
  ```

---

## Setup de Automação

### 1. Cron Jobs

**Editar crontab:**
```bash
crontab -e
```

**Adicionar:**
```cron
# NBA Predictor - Automation
# Recalibração diária (6:00 AM)
0 6 * * * cd /home/user/nba-predictor && /usr/bin/python3 scripts/recalibrate_model.py >> logs/recalibration.log 2>&1

# Dashboard update (6:05 AM)
5 6 * * * cd /home/user/nba-predictor && /usr/bin/python3 monitoring/update_dashboard.py >> logs/monitoring.log 2>&1

# Daily check (6:10 AM)
10 6 * * * cd /home/user/nba-predictor && /usr/bin/python3 monitoring/daily_check.py >> logs/daily_check.log 2>&1

# Backup semanal (Domingo 2:00 AM)
0 2 * * 0 cd /home/user/nba-predictor && tar -czf backups/models_backup_$(date +\%Y\%m\%d).tar.gz models/*.joblib models/*.pkl >> logs/backup.log 2>&1

# Cleanup logs (1º do mês 3:00 AM)
0 3 1 * * find /home/user/nba-predictor/logs -name '*.log' -mtime +30 -delete >> logs/cleanup.log 2>&1
```

**Verificar:**
```bash
crontab -l | grep "NBA Predictor"
```

- [ ] Cron jobs configurados
- [ ] Paths ajustados corretamente
- [ ] Logs directory criado

### 2. Diretórios Necessários

```bash
mkdir -p logs
mkdir -p backups
mkdir -p reports
mkdir -p monitoring
mkdir -p models
```

- [ ] Todos os diretórios criados

---

## Treinamento Inicial

### 1. Modelo ML

- [ ] Train modelo com domain features
  ```bash
  python scripts/quick_retrain.py
  ```
- [ ] Validar modelo salvo
  ```bash
  ls -lh models/ml_model.joblib
  ```

### 2. Calibrator

- [ ] Train calibrator inicial
  ```bash
  python scripts/recalibrate_model.py --lookback-days 60 --min-samples 30
  ```
- [ ] Validar calibrator salvo
  ```bash
  ls -lh models/calibrator.pkl
  ```

---

## Testes de Produção

### 1. Predictions

- [ ] Gerar predictions teste
  ```bash
  python ml_pipeline/predict.py
  ```
- [ ] Verificar calibrator aplicado (check logs)
- [ ] Validar formato de output

### 2. Monitoring

- [ ] Executar daily check manual
  ```bash
  python monitoring/daily_check.py
  ```
- [ ] Verificar dashboard gerado
  ```bash
  ls -lh monitoring/dashboard.html
  ```
- [ ] Abrir dashboard no browser
- [ ] Verificar alertas (se houver)
  ```bash
  tail logs/alerts.log
  ```

### 3. Smoke Tests Finais

- [ ] Todos os 5 testes passando
  ```bash
  python tests/smoke_tests.py
  # Esperado: 5/5 PASS
  ```

---

## Monitoramento (Primeiros 7 Dias)

### Daily Checklist

**Todos os dias às 7:00 AM (após cron):**

- [ ] **Dia 1:** Verificar recalibração rodou
  ```bash
  tail -20 logs/recalibration.log
  ```
- [ ] **Dia 1:** Verificar dashboard atualizado
  ```bash
  ls -lt monitoring/dashboard.html
  ```
- [ ] **Dia 1:** Revisar alertas
  ```bash
  tail logs/alerts.log
  ```
- [ ] **Dia 1:** Verificar métricas
  ```bash
  cat monitoring/daily_metrics.json | tail -20
  ```

**Repetir para Dias 2-7**

### Métricas a Monitorar

**Thresholds Críticos:**
- ECE < 0.08 ✅
- Brier Score < 0.30 ✅
- Accuracy > 52% ✅
- Predictions/dia ≥ 5 ✅

**Se threshold violado:**
1. Check logs para erros
2. Validar data quality
3. Consider retrain se persistir

---

## Relatório Semanal (Domingo)

- [ ] **Domingo Dia 7:** Revisar relatório semanal
  ```bash
  cat monitoring/weekly_report.json
  ```
- [ ] Analisar trends
- [ ] Documentar issues (se houver)
- [ ] Decidir ações corretivas

---

## Troubleshooting Comum

### Calibrator não carrega
```bash
# Verificar
ls -lh models/calibrator.pkl

# Se não existe, treinar
python scripts/recalibrate_model.py --lookback-days 60
```

### Cron não roda
```bash
# Verificar cron status
systemctl status cron  # Linux
# ou
service cron status

# Ver logs cron
grep CRON /var/log/syslog
```

### Dashboard não atualiza
```bash
# Rodar manual
python monitoring/dashboard_generator.py

# Verificar permissões
ls -l monitoring/dashboard.html
```

### Database locks
```bash
# Verificar connection pool
python -c "from utils.connection_pool import ConnectionPool; pool = ConnectionPool(); print(pool.get_stats())"
```

---

## Rollback Plan

**Se algo der errado:**

1. **Stop cron jobs:**
   ```bash
   crontab -e
   # Comentar linhas do NBA Predictor
   ```

2. **Restore modelo:**
   ```bash
   cp models/ml_model_backup_TIMESTAMP.joblib models/ml_model.joblib
   ```

3. **Restore code:**
   ```bash
   git checkout HEAD~1 ml_pipeline/predict.py
   ```

4. **Restore database (se necessário):**
   ```bash
   # Usar backup do database
   ```

---

## Post-Deployment (Após 7 Dias)

- [ ] Análise de resultados
- [ ] Accuracy >= baseline validado
- [ ] ECE < 0.05 confirmado
- [ ] Zero crashes/errors críticos
- [ ] Decisão: Continue ou ajustes

---

## Next Steps (2-4 Semanas)

### Implementar P3.1: Confidence Kelly
- [ ] Implementar TODOs em `betting/confidence_kelly.py`
- [ ] Backtesting
- [ ] Integration

### Completar P2.2 Placeholders
- [ ] Injury impact (priority)
- [ ] Travel fatigue
- [ ] Schedule density

### Expand
- [ ] Spread model
- [ ] Totals model
- [ ] Player props

---

## Aprovação Final

- [ ] **Tech Lead:** Sistema testado e aprovado
- [ ] **Stakeholder:** ROI expectations alinhados
- [ ] **Ops:** Monitoring configurado
- [ ] **Go/No-Go:** ✅ **GO PARA PRODUCTION**

---

**Data de Deploy:** _____________  
**Deployed por:** _____________  
**Status:** _____________  

**Notas:**
- Manter este checklist atualizado
- Documentar quaisquer desvios
- Comunicar issues imediatamente
