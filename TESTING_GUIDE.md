# Guia de Testes - Sistema de Odds Refatorado

## 🧪 Como Testar

### Teste Rápido (Automatizado) ⚡

Execute o script de teste completo que valida todos os componentes:

```bash
cd ~/nba-predictor
python tests/test_odds_refactor.py
```

**O que é testado:**

- ✅ Eliminação de valores fixos (1.90)
- ✅ Normalização de nomes de jogadores
- ✅ Integrity logger e validação de range
- ✅ Player Props scraper (estrutura)
- ✅ OddsDataManager integration
- ✅ OddsProvider interface

**Saída esperada:**

```
TOTAL: 6/6 testes passaram (100.0%)
🎉 TODOS OS TESTES PASSARAM! Sistema pronto para produção.
```

---

### Teste Visual (Player Props) 👁️

Para VER o scraper funcionando em tempo real:

```bash
python scripts/test_player_props_manual.py
```

Escolha **opção 1** para modo visual (navegador abrirá).

**O que acontece:**

1. Navegador Chrome/Chromium abre
2. Acessa Action Network
3. Intercepta dados da API
4. Mostra props encontrados

**Saída esperada:**

```
📊 Primeiros 10 props:

 1. LeBron James        | points   | Line:  25.5 | Over: 1.909 | Under: 1.909
 2. Stephen Curry       | threes   | Line:   4.5 | Over: 1.833 | Under: 1.952
 ...

📈 Distribuição por tipo de prop:
  points      :  45 █████████
  rebounds    :  32 ██████
  assists     :  28 █████
```

---

### Teste Individual (Componentes) 🔬

#### 1. Testar Eliminação de 1.90

```python
from data.scrapers.odds_scraper import get_odds_for_game
from exceptions.odds_exceptions import OddsUnavailableError

try:
    odds = get_odds_for_game("Lakers", "Celtics", odds_cache={})
    print("❌ ERRO: Deveria ter lançado exceção!")
except OddsUnavailableError as e:
    print(f"✅ CORRETO: {e}")
```

**Resultado esperado:** Exceção lançada ✅

---

#### 2. Testar Normalização de Nomes

```python
from data.scrapers.player_name_normalizer import normalize_player_name

# Testes
print(normalize_player_name("LeBron James"))    # → "LeBron James"
print(normalize_player_name("Lebron james"))    # → "LeBron James"
print(normalize_player_name("L. James"))        # → "LeBron James" (fuzzy)
print(normalize_player_name("Fake Player"))     # → None
```

**Resultado esperado:** Normalização correta ✅

---

#### 3. Testar Validação de Range

```python
from data.utils.integrity_logger import validate_odds_range

print(validate_odds_range(0.50, "test", "game1"))  # False (< 1.01)
print(validate_odds_range(1.90, "test", "game2"))  # False (valor fixo)
print(validate_odds_range(2.50, "test", "game3"))  # True (válido)
print(validate_odds_range(100.0, "test", "game4")) # False (> 50.0)
```

**Resultado esperado:** Validação correta + logs em `logs/data_integrity.log` ✅

---

#### 4. Testar Player Props (Produção)

```python
import asyncio
from data.odds_manager import OddsDataManager

async def test():
    manager = OddsDataManager()
    props = await manager.fetch_player_props("2024-12-18")
    
    print(f"Props encontrados: {len(props)}")
    
    for prop in props[:5]:
        print(f"{prop.player_name} - {prop.prop_type}: {prop.line}")

asyncio.run(test())
```

**Resultado esperado:** Lista de props ✅

---

## 📋 Checklist de Validação

Use este checklist para validar manualmente:

### ✅ Eliminação de 1.90

- [ ] `odds_scraper.py` lança `OddsUnavailableError` quando sem cache
- [ ] `odds_web_scraper.py` não retorna 1.90 no JSON-LD
- [ ] `oddspedia_provider.py` pula jogos sem odds reais
- [ ] Nenhum arquivo contém `default=1.90` ou `return {'home_odds': 1.90`

### ✅ Player Props

- [ ] `action_network_scraper.py` existe e roda sem erros
- [ ] `PlayerProp` dataclass funciona
- [ ] `OddsDataManager.fetch_player_props()` retorna lista
- [ ] Props têm nomes normalizados (sem "L. James" etc)

### ✅ Validação e Logging

- [ ] `integrity_logger.py` existe
- [ ] `logs/data_integrity.log` é criado ao validar odds
- [ ] Odds < 1.01 são rejeitadas
- [ ] Odds > 50.0 são rejeitadas
- [ ] Odds == 1.90 são detectadas como fixas

### ✅ Integração

- [ ] `OddsProvider` tem método `get_player_props()`
- [ ] Imports funcionam sem circular dependency
- [ ] Sistema não quebra funcionamento existente

---

## 🐛 Troubleshooting

### Erro: "Playwright not installed"

```bash
pip install playwright
playwright install chromium
```

### Erro: "Module 'data.utils.integrity_logger' not found"

Verifique se o arquivo foi criado:

```bash
ls -la ~/nba-predictor/data/utils/integrity_logger.py
```

Se não existir, crie o diretório primeiro:

```bash
mkdir -p ~/nba-predictor/data/utils/
```

### Erro: "Cannot normalize player name"

Verifique se `data/nba_player_stats.csv` existe:

```bash
ls -la ~/nba-predictor/data/nba_player_stats.csv
```

### Nenhum prop encontrado

Possíveis causas:

1. Não há jogos NBA hoje → Teste com data de jogo conhecida
2. Action Network bloqueou → Instale `playwright-stealth`
3. API mudou → Verifique browser inspection novamente

---

## 📊 Validação em Produção

### 1. Monitorar Logs

```bash
# Ver logs de integridade em tempo real
tail -f ~/nba-predictor/logs/data_integrity.log

# Ver últimas 50 linhas
tail -50 ~/nba-predictor/logs/data_integrity.log
```

### 2. Verificar Taxa de Sucesso

```python
from data.odds_manager import OddsDataManager
import asyncio

async def check_stats():
    manager = OddsDataManager()
    stats = manager.get_stats()
    print("Estatísticas dos providers:")
    for provider, rate in stats['success_rates'].items():
        print(f"  {provider}: {rate}% sucesso")

asyncio.run(check_stats())
```

### 3. Health Check

```python
from data.odds_manager import OddsDataManager
import asyncio

async def health():
    manager = OddsDataManager()
    results = await manager.health_check_all()
    
    for provider, is_healthy in results.items():
        status = "✅" if is_healthy else "❌"
        print(f"{status} {provider}")

asyncio.run(health())
```

---

## 🚀 Próximos Passos

Após validar que tudo funciona:

1. **Instalar playwright-stealth** (opcional, mas recomendado):

   ```bash
   pip install playwright-stealth
   ```

2. **Aplicar validação em todos scrapers** (atualmente só oddspedia):
   - Editar `sbr_scraper.py`
   - Editar `action_network_scraper.py`
   - Adicionar `validate_odds_range()` antes de criar GameOdds

3. **Deploy em staging** e monitorar por 48h

4. **Deploy em produção** se tudo OK

---

## 📞 Suporte

Se algo não funcionar:

1. Execute o teste automatizado: `python tests/test_odds_refactor.py`
2. Verifique os logs: `tail -50 logs/data_integrity.log`
3. Verifique o surgery_report.md para detalhes técnicos

**Bons testes!** 🎉
