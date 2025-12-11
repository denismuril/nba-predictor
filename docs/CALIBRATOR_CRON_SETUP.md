# ⏰ Agendamento Automático do Calibrador

## ✅ Configuração do Cron

O calibrador está configurado para **treinar automaticamente** toda segunda-feira às 09:00.

### 📅 Cronograma Atual

```bash
# Treinar calibrador semanalmente (toda segunda às 09:00)
0 9 * * 1 cd /home/denis/nba-predictor && python scripts/train_calibrator.py >> logs/calibrator_training.log 2>&1
```

**Tradução:**

- `0 9 * * 1` = Toda segunda-feira (1) às 09:00 (hora 9, minuto 0)
- `>>` = Salva logs em `logs/calibrator_training.log`
- `2>&1` = Captura erros também

---

## 📊 Logs e Monitoramento

### Ver logs do último treinamento

```bash
tail -50 logs/calibrator_training.log
```

### Ver apenas resultados (filtrar INFO)

```bash
grep "Resultados do Treinamento" logs/calibrator_training.log -A 5
```

### Monitorar em tempo real (se rodar manualmente)

```bash
tail -f logs/calibrator_training.log
```

---

## 🔄 Cronograma Completo do Sistema

```bash
# Atualização principal (diária às 10:00)
0 10 * * * /bin/bash /home/denis/nba-predictor/scripts/orchestrator.sh

# Bot Telegram (ao reiniciar)
@reboot /bin/bash /home/denis/nba-predictor/scripts/run_bot.sh >> logs/cron_bot.log 2>&1

# Atualização RAPM (diária às 09:30)
30 9 * * * cd /home/denis/nba-predictor && python scripts/update_rapm.py >> logs/rapm_updates.log 2>&1

# Treinar calibrador (toda segunda às 09:00) ✨ NOVO
0 9 * * 1 cd /home/denis/nba-predictor && python scripts/train_calibrator.py >> logs/calibrator_training.log 2>&1
```

---

## ⚙️ Modificar Agendamento

### Treinar diariamente ao invés de semanalmente

```bash
# Editar crontab
crontab -e

# Alterar de:
0 9 * * 1  # (apenas segunda)

# Para:
0 9 * * *  # (todos os dias)
```

### Treinar a cada 3 dias

```bash
0 9 */3 * *
```

### Verificar crontab atual

```bash
crontab -l
```

---

## 🧪 Testar Manualmente

Para testar o treinamento manualmente (sem esperar a segunda-feira):

```bash
python scripts/train_calibrator.py
```

Ou com logs:

```bash
python scripts/train_calibrator.py >> logs/calibrator_training.log 2>&1
```

---

## 📈 Recomendações de Frequência

| Período | Frequência Recomendada | Razão |
|---------|------------------------|-------|
| **Temporada Regular** | Semanal (atual) | Padrões de jogo mais estáveis |
| **Playoffs** | A cada 3 dias | Mais variabilidade, times evoluem rápido |
| **All-Star Break** | Pausar | Poucos jogos, dados inconsistentes |
| **Início da Temporada** | A cada 2 dias | Modelo se ajustando a nova temporada |

---

## ✅ Verificação Rápida

Para confirmar que está funcionando:

```bash
# Ver próxima execução
crontab -l | grep calibrator

# Verificar se o log existe (após primeira execução)
ls -lh logs/calibrator_training.log

# Ver última linha do log
tail -1 logs/calibrator_training.log
```
