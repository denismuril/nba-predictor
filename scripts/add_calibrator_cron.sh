#!/bin/bash
# Script temporário para adicionar calibrator ao crontab

# Salvar crontab atual
crontab -l > /tmp/current_cron

# Adicionar nova linha para treinamento do calibrador (toda segunda às 9h)
echo "# Treinar calibrador semanalmente (toda segunda às 09:00)" >> /tmp/current_cron
echo "0 9 * * 1 cd /home/denis/nba-predictor && python scripts/train_calibrator.py >> logs/calibrator_training.log 2>&1" >> /tmp/current_cron

# Instalar novo crontab
crontab /tmp/current_cron

# Mostrar resultado
echo "✅ Crontab atualizado:"
crontab -l

# Limpar
rm /tmp/current_cron
