# Odds Scraper - Guia de Configuração

## TheOddsAPI Setup

### 1. Obter API Key

1. Acesse: https://the-odds-api.com/
2. Clique em "Get API Key" (gratuito para 500 requests/mês)
3. Copie sua API key

### 2. Configurar no .env

Adicione a seguinte linha ao arquivo `.env`:

```bash
ODDS_API_KEY=your_api_key_here
```

### 3. Testar Configuração

```bash
# Rodar teste de integração (requer API key válida)
python -c "from data.scrapers.odds_scraper import obter_odds; print(obter_odds())"
```

## Hierarquia de Fallback

O sistema tenta buscar odds na seguinte ordem:

1. **TheOddsAPI** (recomendado)
   - Requer: `ODDS_API_KEY` no .env
   - Quota: 500 requests/mês (grátis)
   - Cobertura: Todas as casas de apostas principais

2. **Odds Shark** (backup - TODO)
   - Web scraping
   - Sem limites de quota
   - Pode quebrar se site mudar

3. **Default (1.90)** (último recurso)
   - ⚠️ WARNING: Kelly Criterion será impreciso
   - ⚠️ Expected Value não será confiável

## Exemplo de Uso

```python
from data.scrapers.odds_scraper import obter_odds, get_odds_for_game

# Buscar odds de todos os jogos do dia
odds_cache = obter_odds()

# Obter odds de um jogo específico
game_odds = get_odds_for_game('Lakers', 'Warriors', odds_cache=odds_cache)

print(f"Lakers: {game_odds['home_odds']}")
print(f"Warriors: {game_odds['away_odds']}")
print(f"Source: {game_odds['source']}")
```

## Validação de Odds

O sistema valida automaticamente:
- ✅ Odds entre 1.01 e 50.0
- ✅ Soma de probabilidades implícitas > 1.0 (vigorish)
- ✅ Vigorish < 30% (odds realistas)

Odds inválidos são removidos e logados.

## Monitoramento

Verifique logs para:
```
✅ Odds obtidos via TheOddsAPI: 12 jogos
⚠️  3/15 jogos com odds inválidos removidos
🚨 ATENÇÃO: Nenhuma fonte de odds disponível! Usando odds padrão...
```

## Troubleshooting

### "ODDS_API_KEY não configurado"
- Solução: Adicionar `ODDS_API_KEY=...` ao arquivo .env

### "TheOddsAPI quota exceeded"
- Solução: Aguardar reset mensal ou fazer upgrade do plano
- Fallback: Sistema usará default (1.90) com warning

### "Requests restantes hoje: 0"
- Quota esgotada
- Sistema automaticamente usa fallback

## Integração com Sistema

O odds scraper é chamado automaticamente em:
- `main.py` - Ao gerar predições
- `ml_pipeline/predict.py` - Ao calcular Expected Value
- `interfaces/cli.py` - Ao exibir Kelly Criterion

Nenhuma mudança de código necessária! 🎉
