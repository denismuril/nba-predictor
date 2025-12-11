# Arquivos Necessários para NBA Predictor

## 📂 Diretório: `data/`

Para o sistema funcionar corretamente, você precisa copiar os seguintes arquivos de `C:\Projetos-Compartilhados\NBA` para `\\wsl$\Ubuntu\home\denis\nba-predictor\data\`:

### Arquivos Obrigatórios:

1. **bpm_2025.xlsx** - Estatísticas de BPM (Box Plus/Minus) dos jogadores
2. **rapm_2025.xlsx** - Estatísticas de RAPM (Regularized Adjusted Plus-Minus)
3. **lebron_2025.xlsx** - Estatísticas LEBRON dos jogadores
4. **pie_2025.xlsx** - Estatísticas PIE (Player Impact Estimate)

### Como Copiar:

**Opção 1: Via Windows Explorer**
1. Abra o Windows Explorer
2. Navegue até: `\\wsl$\Ubuntu\home\denis\nba-predictor\data\`
3. Copie os 4 arquivos de `C:\Projetos-Compartilhados\NBA` para este diretório

**Opção 2: Via PowerShell**
```powershell
Copy-Item "C:\Projetos-Compartilhados\NBA\bpm_2025.xlsx" "\\wsl$\Ubuntu\home\denis\nba-predictor\data\"
Copy-Item "C:\Projetos-Compartilhados\NBA\rapm_2025.xlsx" "\\wsl$\Ubuntu\home\denis\nba-predictor\data\"
Copy-Item "C:\Projetos-Compartilhados\NBA\lebron_2025.xlsx" "\\wsl$\Ubuntu\home\denis\nba-predictor\data\"
Copy-Item "C:\Projetos-Compartilhados\NBA\pie_2025.xlsx" "\\wsl$\Ubuntu\home\denis\nba-predictor\data\"
```

### Verificação:

Após copiar, execute no PowerShell:
```powershell
Get-ChildItem "\\wsl$\Ubuntu\home\denis\nba-predictor\data\*.xlsx"
```

Você deve ver os 4 arquivos listados.

---

## 📝 Estrutura de Dados Esperada

### bpm_2025.xlsx
- **Sheet**: "Worksheet" (ou primeira sheet)
- **Colunas esperadas**: Player, Tm (Team), BPM, OBPM, DBPM, MP (Minutes Played)

### rapm_2025.xlsx
- **Sheet**: "Planilha3" (ou primeira sheet)
- **Colunas esperadas**: team, time decay orapm, time decay drapm

### lebron_2025.xlsx
- **Sheet**: Primeira sheet
- **Colunas esperadas**: Team, O-LEBRON, D-LEBRON

### pie_2025.xlsx
- **Sheet**: Primeira sheet
- **Colunas esperadas**: Team, PIE

---

## ⚠️ Notas Importantes

1. Os nomes dos arquivos devem ser **exatamente** como listado acima (case-sensitive no Linux)
2. Se os arquivos tiverem nomes diferentes em `C:\Projetos-Compartilhados\NBA`, renomeie-os ao copiar
3. Certifique-se de que os arquivos não estão corrompidos ou protegidos por senha
4. O sistema tentará ler estes arquivos ao iniciar - se não encontrar, usará APIs como fallback

---

## 🔄 Atualização dos Dados

Para atualizar as estatísticas:
1. Substitua os arquivos no diretório `data/`
2. Execute novamente o predictor: `python3 nba_predictor.py YYYY-MM-DD`
