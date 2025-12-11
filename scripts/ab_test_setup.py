import pandas as pd
import joblib
import logging
import json
import os
from datetime import datetime
from ml_pipeline.data_preparation import load_historical_data, add_rolling_features, add_advanced_features
from ml_pipeline.predict import load_model as load_production_model

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AB_LOG_FILE = 'data/monitoring/ab_test_log.json'

def setup_ab_test():
    """
    Configura e executa uma rodada de teste A/B.
    Compara:
    - Modelo A: Produção (Ensemble Calibrado)
    - Modelo B: Challenger (Ensemble Blending - se existir, ou outro experimental)
    """
    logger.info("⚖️ Iniciando Rodada de Teste A/B...")
    
    # 1. Carregar Dados Recentes (Jogos de Hoje/Amanhã)
    # Para simulação, vamos pegar os últimos jogos disponíveis no DB
    df = load_historical_data(seasons=['2025-26'], apply_weights=False)
    df = add_rolling_features(df)
    df = add_advanced_features(df)
    
    # Pegar últimos 5 jogos para teste
    df_test = df.tail(5).copy()
    
    if df_test.empty:
        logger.warning("⚠️ Sem jogos recentes para teste A/B.")
        return

    # 2. Carregar Modelo A (Produção)
    try:
        model_a = joblib.load('data/models/ensemble_model_calibrated_isotonic.joblib')
        features_a = joblib.load('data/models/feature_names_final.joblib')
        logger.info("✅ Modelo A (Produção) carregado.")
    except:
        logger.error("❌ Falha ao carregar Modelo A.")
        return

    # 3. Carregar Modelo B (Challenger)
    # Tentar carregar Blending, senão usar Baseline
    try:
        # Blending requer carregamento especial (base models + meta model)
        # Simplificação: Vamos assumir que existe um 'blending_meta_model.joblib' que sabe se virar
        # Na prática, precisaríamos de uma classe wrapper.
        # Vamos usar um placeholder ou tentar carregar o blending se implementado corretamente.
        
        # Se blending_base_models.joblib existir, é um sinal
        if os.path.exists('data/models/blending_meta_model.joblib'):
             logger.info("✅ Modelo B (Blending) detectado. (Lógica de inferência complexa pendente)")
             # TODO: Implementar inferência completa do Blending
             model_b = None 
             model_b_name = "Blending (Placeholder)"
        else:
             # Fallback: Usar modelo não calibrado como Challenger
             model_b = joblib.load('data/models/ensemble_model_final.joblib')
             features_b = features_a # Assume mesmas features
             model_b_name = "Ensemble Não-Calibrado"
             logger.info(f"✅ Modelo B ({model_b_name}) carregado.")
             
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar Modelo B: {e}")
        model_b = None
        model_b_name = "None"

    # 4. Executar Previsões
    results = []
    
    for idx, row in df_test.iterrows():
        game_id = f"{row['date']}_{row['home_team']}_{row['away_team']}"
        
        # Prep input A
        X_a = pd.DataFrame([row[features_a]])
        prob_a = model_a.predict_proba(X_a)[0][1]
        
        # Prep input B
        prob_b = 0.5
        if model_b:
            try:
                X_b = pd.DataFrame([row[features_b]])
                prob_b = model_b.predict_proba(X_b)[0][1]
            except:
                pass
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'winner_actual': int(row['winner']),
            'model_a_prob': float(prob_a),
            'model_a_pred': int(prob_a >= 0.5),
            'model_b_name': model_b_name,
            'model_b_prob': float(prob_b),
            'model_b_pred': int(prob_b >= 0.5)
        }
        results.append(entry)
        
    # 5. Salvar Log
    with open(AB_LOG_FILE, 'a') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
            
    logger.info(f"💾 {len(results)} registros de teste A/B salvos em {AB_LOG_FILE}")
    
    # Exibir resumo
    df_res = pd.DataFrame(results)
    print("\n📊 Resumo Teste A/B (Últimos Jogos):")
    print(df_res[['home_team', 'away_team', 'winner_actual', 'model_a_prob', 'model_b_prob']])

if __name__ == "__main__":
    setup_ab_test()
