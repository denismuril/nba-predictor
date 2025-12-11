# Guia de Treinamento e Validação

## 🎓 Passo a Passo Completo

### 1️⃣ Treinar o Calibrador

Execute o script de treinamento do calibrador com dados históricos:

```bash
python scripts/train_calibrator.py
```

**Opções disponíveis:**

- `--lookback 60` - Número de dias de histórico (padrão: 60)
- `--min-samples 50` - Mínimo de jogos necessários (padrão: 50)

**Exemplo com parâmetros customizados:**

```bash
python scripts/train_calibrator.py --lookback 90 --min-samples 100
```

**O que o script faz:**

1. Carrega o modelo treinado
2. Busca jogos históricos dos últimos N dias
3. Gera features para esses jogos
4. Calcula previsões vs resultados reais
5. Treina o calibrador usando Isotonic Regression
6. Salva em `data/models/calibrator.pkl`

**Output esperado:**

```
📊 Resultados do Treinamento:
   Jogos usados: 150
   Brier Score:  0.2345 → 0.2123 (+9.5%)
   Log Loss:     0.5678 → 0.5234 (+7.8%)
   ECE:          0.0856 → 0.0423 (+50.6%)

✅ Calibrador salvo em: data/models/calibrator.pkl
```

---

### 2️⃣ Validar o Pipeline Completo

Depois de treinar o calibrador, valide que tudo funciona:

```bash
python scripts/validate_pipeline.py
```

**O que o script faz:**

1. Testa `predict_next_games()` com os jogos de hoje
2. Valida que todas as colunas novas existem
3. Compara probabilidades raw vs calibradas
4. Mostra distribuição de confiança (HIGH/MEDIUM/LOW)

**Output esperado:**

```
🧪 VALIDAÇÃO COMPLETA DO PIPELINE DE PREDIÇÃO

📊 Testando predict_next_games()...
✅ Todas as colunas esperadas presentes

📈 RESULTADOS (3 jogos):

🏀 LAL vs BOS
   Prob Raw:    0.623
   Prob Calib:  0.587
   Diferença:   -0.036
   Confiança:   HIGH (0.853)
   IC 95%:      [0.512, 0.662]

...

✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO
```

---

### 3️⃣ Testar Previsões em Tempo Real

Para testar com jogos de hoje:

```bash
# Atualizar resultados mais recentes
python scripts/update_game_results.py

# Gerar previsões
python ml_pipeline/predict.py
```

---

## ⚠️ Troubleshooting

### Erro: "Apenas X jogos disponíveis. Mínimo necessário: 50"

**Solução**: Reduza o `min_samples` ou aumente `lookback_days`

```bash
python scripts/train_calibrator.py --lookback 90 --min-samples 30
```

### Erro: "Calibrator não foi fitted! Retornando probabilidades raw"

**Causa**: O calibrador não foi treinado ainda
**Solução**: Execute `train_calibrator.py` primeiro

### Erro: Feature names incompatíveis

**Causa**: Modelo foi retreinado mas calibrador usa features antigas
**Solução**: Retreine o calibrador após retreinar o modelo

---

## 📊 Métricas de Avaliação

### Brier Score

- **Range**: 0-1 (menor é melhor)
- **Idealmente**: < 0.20
- Mede a acurácia das probabilidades

### Expected Calibration Error (ECE)

- **Range**: 0-1 (menor é melhor)  
- **Idealmente**: < 0.05
- Mede o alinhamento entre probabilidades preditas e frequências reais

### Log Loss

- **Range**: 0-∞ (menor é melhor)
- **Idealmente**: < 0.55
- Penaliza previsões confiantes incorretas

---

## 🔄 Retreinamento Periódico

Recomenda-se retreinar o calibrador:

- **Semanalmente** durante a temporada regular
- **A cada 3 dias** durante playoffs (mais variabilidade)

```bash
# Cron job sugerido (diário às 10AM)
0 10 * * * cd /home/denis/nba-predictor && python scripts/train_calibrator.py
```
