import pandas as pd
import numpy as np
import logging
import os
import sys
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Adicionar root ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.train_spread_real import load_historical_data, prepare_features_v2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_walk_forward_validation(start_date=None, end_date=None, step_days=7):
    """
    Executa validação Walk-Forward simulando a temporada.
    Treina -> Prevê -> Avança -> Repete.
    
    Se start_date e end_date não forem fornecidos, usa a temporada atual (2025-26)
    desde o início (21 de outubro de 2025) até a data atual.
    Atualiza automaticamente toda semana com novos jogos.
    """
    # Definir datas dinamicamente para temporada atual se não fornecidas
    if start_date is None:
        start_date = '2025-10-21'  # Início da temporada 2025-26
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')  # Data atual
    
    logger.info(f"🚀 Iniciando Walk-Forward Validation ({start_date} a {end_date})...")
    
    # 1. Carregar todos os dados
    df_all = load_historical_data()
    if df_all is None or df_all.empty:
        logger.error("❌ Sem dados para validação.")
        return

    # Garantir ordenação temporal
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date').reset_index(drop=True)
    
    # Features
    X, y = prepare_features_v2(df_all)
    # Adicionar coluna de data para filtro
    X['date'] = df_all['date']
    X['point_differential'] = y # Target real para avaliação
    
    current_date = pd.to_datetime(start_date)
    final_date = pd.to_datetime(end_date)
    
    results = []
    capital = 1000.0 # Banca inicial
    capital_history = [{'date': current_date, 'capital': capital}]
    
    # Carregar hiperparâmetros otimizados do Optuna
    PARAMS_FILE = os.path.join("data", "models", "best_hyperparameters.joblib")
    
    if os.path.exists(PARAMS_FILE):
        logger.info(f"💎 Carregando hiperparâmetros OTIMIZADOS de {PARAMS_FILE}...")
        params = joblib.load(PARAMS_FILE)
        params['n_jobs'] = -1
        params['random_state'] = 42
        params['objective'] = 'reg:absoluteerror'
    else:
        logger.warning("⚠️  Arquivo de hiperparâmetros não encontrado. Usando DEFAULT.")
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': 42,
            'objective': 'reg:absoluteerror'
        }
    
    while current_date < final_date:
        next_date = current_date + timedelta(days=step_days)
        logger.info(f"📅 Simulando semana: {current_date.date()} a {next_date.date()}")
        
        # 1. Split Treino / Teste
        train_mask = X['date'] < current_date
        test_mask = (X['date'] >= current_date) & (X['date'] < next_date)
        
        X_train_full = X[train_mask]
        X_test_full = X[test_mask]
        
        if X_test_full.empty:
            logger.warning("   ⚠️ Sem jogos nesta semana. Pulando.")
            current_date = next_date
            continue
            
        # Remover colunas auxiliares para treino
        features_cols = [c for c in X.columns if c not in ['date', 'point_differential']]
        
        X_train = X_train_full[features_cols]
        y_train = X_train_full['point_differential'] # Usando a coluna que salvamos no X temporariamente
        
        X_test = X_test_full[features_cols]
        y_test_real = X_test_full['point_differential']
        
        # 2. Treinar Modelo (Re-treino semanal)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # 3. Prever
        preds = model.predict(X_test)
        
        # 4. Avaliar P&L (Simulação de Aposta)
        # Estratégia Simples: Se Modelo diz > 0 e Real > 0 (Win)
        # Assumindo Odds 1.90 e Stake Fixa 1%
        
        weekly_pnl = 0
        wins = 0
        losses = 0
        
        for pred, real in zip(preds, y_test_real):
            # Aposta no Spread (Margem)
            # Se modelo prevê Home +5, e Real foi Home +2 -> Erro de 3.
            # Para P&L, precisamos da Linha de Vegas. Como não temos histórico de linhas aqui,
            # vamos simular que a Linha de Vegas foi "Perfeita" (0 de margem média) ou usar Pick'em.
            # Simulação: Apostar no Vencedor (Moneyline) se confiança alta
            
            # Se pred > 3 (Home vence por 3+), aposta Home
            # Se pred < -3 (Away vence por 3+), aposta Away
            
            stake = 10.0 # 1% de 1000
            bet_result = 0.0
            
            if pred > 2.0: # Aposta Home
                if real > 0: # Home venceu
                    bet_result = stake * 0.90 # Lucro (Odds 1.90)
                    wins += 1
                else:
                    bet_result = -stake
                    losses += 1
            elif pred < -2.0: # Aposta Away
                if real < 0: # Away venceu
                    bet_result = stake * 0.90
                    wins += 1
                else:
                    bet_result = -stake
                    losses += 1
            
            weekly_pnl += bet_result
            
        capital += weekly_pnl
        capital_history.append({'date': next_date, 'capital': capital})
        
        mae = mean_absolute_error(y_test_real, preds)
        logger.info(f"   💰 P&L Semanal: ${weekly_pnl:.2f} | Banca: ${capital:.2f} | MAE: {mae:.2f}")
        
        # Avançar
        current_date = next_date

    # Resultados Finais
    logger.info("="*50)
    logger.info(f"🏁 Resultado Final Walk-Forward")
    logger.info(f"💰 Banca Inicial: $1000.00")
    logger.info(f"💰 Banca Final:   ${capital:.2f}")
    logger.info(f"📈 ROI: {((capital - 1000)/1000)*100:.1f}%")
    logger.info("="*50)
    
    # Salvar gráfico
    df_res = pd.DataFrame(capital_history)
    plt.figure(figsize=(10, 6))
    plt.plot(df_res['date'], df_res['capital'], marker='o')
    plt.title('Curva de Capital - Walk Forward Validation (2023-24)')
    plt.xlabel('Data')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.savefig('results/walk_forward_equity_curve.png')
    logger.info("📊 Gráfico salvo em results/walk_forward_equity_curve.png")

if __name__ == "__main__":
    run_walk_forward_validation()
