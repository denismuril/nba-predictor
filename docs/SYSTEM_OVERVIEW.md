# Visão Geral do Sistema NBA Predictor

## 1. Introdução
O **NBA Predictor** é um sistema sofisticado projetado para prever os resultados dos jogos da NBA usando uma combinação de análise estatística, power ratings, simulações de Monte Carlo e aprendizado de máquina. Ele agrega dados de múltiplas fontes (calendário, lesões, estatísticas de jogadores, odds, árbitros) para fornecer insights de apostas acionáveis.

## 2. Arquitetura do Sistema

O projeto segue uma arquitetura modular para garantir manutenibilidade e escalabilidade.

### Estrutura de Diretórios
```
nba-predictor/
├── config/             # Configurações e constantes
├── core/               # Lógica de negócio principal (algoritmos, simulação)
├── data/               # Camada de dados (banco de dados, scrapers, repositórios)
├── docs/               # Documentação do sistema
├── interfaces/         # Interfaces de usuário (CLI, Web)
├── ml_pipeline/        # Fluxo de trabalho de Machine Learning
├── utils/              # Utilitários auxiliares (logging, exportação)
├── nba_predictor_web.py # Ponto de entrada da Web App Streamlit
└── run_pipeline.sh     # Script de automação
```

## 3. Módulos Principais

### 3.1 Camada de Dados (`data/`)
Responsável por buscar e persistir dados.
- **`repositories/`**: `db_manager.py` gerencia interações com SQLite (salvar jogos, previsões).
- **`scrapers/`**:
    - `schedule_scraper.py`: Busca o calendário de jogos.
    - `injury_scraper.py`: Coleta relatórios de lesões (CBS Sports, ESPN).
    - `stats_scraper.py`: Busca estatísticas de jogadores (DFS, métricas avançadas).
    - `odds_scraper.py`: Busca odds de apostas (TheOddsAPI).
    - `referee_scraper.py`: Busca designações e estatísticas de árbitros.

### 3.2 Lógica Principal (`core/`)
Contém o "cérebro" da operação.
- **`algorithms.py`**: Implementa `calcular_power_rating_v11`. Calcula a força do time com base no Net Rating, lesões, vantagem de jogar em casa e impacto do árbitro.
- **`simulation.py`**: Executa simulações de Monte Carlo (1.000.000 iterações vetorizadas) para determinar probabilidades de vitória baseadas nos power ratings.
- **`roster_manager.py`**: Gerencia elencos de times e disponibilidade de jogadores.

### 3.3 Machine Learning (`ml_pipeline/`)
Melhora as previsões usando dados históricos.
- **`data_preparation.py`**: Prepara features para treinamento.
- **`train_ensemble.py`**: Treina um modelo de ensemble (Random Forest, XGBoost, Gradient Boosting).
- **`predict.py`**: Gera probabilidades baseadas em ML para os próximos jogos.

### 3.4 Interfaces (`interfaces/`)
- **CLI (`interfaces/cli.py`)**: Ferramenta de linha de comando para executar previsões para uma data específica.
- **Web App (`nba_predictor_web.py`)**: Dashboard interativo Streamlit para visualizar previsões, analisar jogos e rastrear apostas.

## 4. Fluxo de Dados

1.  **Entrada**: Usuário seleciona uma data (CLI ou Web).
2.  **Coleta**: Sistema coleta calendário, lesões, estatísticas, odds e dados de árbitros.
3.  **Processamento**:
    - **Power Rating**: Calcula `pr_casa` e `pr_visitante` usando `algorithms.py`.
    - **Simulação**: Executa simulação de Monte Carlo para obter `prob_mc`.
    - **ML**: Executa modelo de ML para obter `prob_ml`.
4.  **Decisão**:
    - **Critério de Kelly**: Compara probabilidades calculadas com odds do mercado para recomendar tamanho da aposta.
    - **Total/Spread**: Estima spread de pontos e total de pontos.
5.  **Saída**: Resultados exibidos na CLI/Web e salvos em CSV/Excel/Banco de Dados.

## 5. Guia de Uso

### Pré-requisitos
- Python 3.10+
- Dependências: `pip install -r requirements.txt`
- Chave TheOddsAPI (no `.env` ou `config/constants.py`)

### Executando a CLI
Para gerar previsões para hoje:
```bash
python interfaces/cli.py
```
Para gerar para uma data específica:
```bash
python interfaces/cli.py --date 2025-11-22
```

### Executando a Web App
Para iniciar o dashboard interativo:
```bash
streamlit run nba_predictor_web.py
```

### Pipeline Automatizado
Para executar o processo diário completo (atualizar DB, treinar ML, prever):
```bash
./run_pipeline.sh
```
