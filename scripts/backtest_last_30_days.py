import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from ml_pipeline.data_preparation import load_historical_data, add_rolling_features, add_advanced_features
from ml_pipeline.feature_selection import select_features
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backtest_last_30_days():
    """
    Realiza backtesting do modelo nos últimos 30 dias de dados reais.
    """
    logger.info("🔄 Iniciando Backtesting dos últimos 30 dias...")

    # 1. Carregar dados históricos RECENTES
    # Carregar temporadas atuais e anterior para ter histórico suficiente para rolling features
    df_raw = load_historical_data(seasons=['2023-24', '2024-25', '2025-26'], apply_weights=False)
    
    # 2. Preparar Features
    logger.info("🛠️ Preparando features...")
    df = add_rolling_features(df_raw)
    df = add_advanced_features(df)
    
    # 3. Filtrar últimos 30 dias
    # Encontrar a data mais recente no dataset
    df['date'] = pd.to_datetime(df['date'])
    max_date = df['date'].max()
    start_date = max_date - timedelta(days=30)
    
    logger.info(f"📅 Período de Backtest: {start_date.date()} a {max_date.date()}")
    
    df_test = df[(df['date'] >= start_date) & (df['date'] <= max_date)].copy()
    
    if df_test.empty:
        logger.error("❌ Nenhum jogo encontrado no período de 30 dias!")
        return

    logger.info(f"📊 Total de jogos para teste: {len(df_test)}")

    # 4. Carregar Modelo e Features
    try:
        model = joblib.load('data/models/ensemble_model_v6.joblib')
        features = joblib.load('data/models/feature_names_v6.joblib')
        logger.info("✅ Modelo calibrado carregado com sucesso.")
    except FileNotFoundError:
        logger.error("❌ Modelo não encontrado! Certifique-se de ter treinado e calibrado o modelo.")
        return

    # 5. Fazer Previsões
    # Garantir que todas as colunas existem (preencher com 0 se faltar alguma - edge case)
    for col in features:
        if col not in df_test.columns:
            # logger.warning(f"⚠️ Feature ausente: {col} (preenchendo com 0)")
            df_test[col] = 0
            
    X_test = df_test[features].copy()
    X_test = X_test.fillna(0) # Garantir sem NaNs
    
    # Converter winner para int (0 ou 1) se estiver como string ('HOME', 'AWAY')
    # Assumindo que 1 = HOME, 0 = AWAY
    if df_test['winner'].dtype == 'object':
        y_true = df_test['winner'].apply(lambda x: 1 if str(x).upper() == 'HOME' else 0).astype(int)
    else:
        y_true = df_test['winner'].astype(int)
    
    # Probabilidades
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Previsões (Threshold 0.5)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # 6. Calcular Métricas
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    
    logger.info("\n" + "="*40)
    logger.info(f"🏆 RESULTADOS BACKTEST (Últimos 30 dias)")
    logger.info("="*40)
    logger.info(f"✅ Accuracy:    {acc:.2%}")
    logger.info(f"📉 Brier Score: {brier:.4f}")
    logger.info(f"📉 Log Loss:    {ll:.4f}")
    logger.info(f"🔢 Jogos:       {len(df_test)}")
    logger.info("="*40)
    
    # 7. Análise de EV (Simulação de Apostas)
    # Assumindo odds médias de 1.90 para simplificação se não tiver odds reais no dataset
    # Se tiver odds reais, usar.
    
    if 'odds_home' in df_test.columns and 'odds_away' in df_test.columns:
        logger.info("\n💰 Análise Financeira (EV Betting)")
        initial_bankroll = 1000
        current_bankroll = initial_bankroll
        bets_placed = 0
        wins = 0
        
        for idx, row in df_test.iterrows():
            prob_home = y_prob[idx - df_test.index[0]] # Ajuste de índice se necessário, mas iterrows é seguro
            # Na verdade, y_prob é array numpy, precisamos alinhar.
            # Melhor usar zip
            pass
        
        # Refazer loop com zip para garantir alinhamento
        profit = 0
        roi_bets = 0
        
        for true_outcome, prob, odd_h, odd_a in zip(y_true, y_prob, df_test['odds_home'], df_test['odds_away']):
            # Estratégia: Apostar se EV > 5%
            
            # Home
            ev_home = (prob * odd_h) - 1
            # Away
            prob_away = 1 - prob
            ev_away = (prob_away * odd_a) - 1
            
            bet_amount = 50 # Aposta fixa $50
            
            if ev_home > 0.05:
                bets_placed += 1
                roi_bets += bet_amount
                if true_outcome == 1: # Home venceu
                    profit += bet_amount * (odd_h - 1)
                    wins += 1
                else:
                    profit -= bet_amount
            
            elif ev_away > 0.05:
                bets_placed += 1
                roi_bets += bet_amount
                if true_outcome == 0: # Away venceu
                    profit += bet_amount * (odd_a - 1)
                    wins += 1
                else:
                    profit -= bet_amount
                    
        roi = (profit / roi_bets) * 100 if roi_bets > 0 else 0
        
        logger.info(f"💵 Apostas Feitas: {bets_placed}")
        logger.info(f"✅ Vitórias:       {wins} ({wins/bets_placed:.1%} win rate)" if bets_placed > 0 else "✅ Vitórias: 0")
        logger.info(f"📈 Lucro Líquido:  ${profit:.2f}")
        logger.info(f"📊 ROI:            {roi:.2f}%")
        logger.info("="*40)

if __name__ == "__main__":
    backtest_last_30_days()
