"""
Confidence Intervals - Adiciona incerteza às previsões

Calcula intervalos de confiança para probabilidades e spreads.
"""
import numpy as np
from scipy import stats

def calculate_confidence_interval(predictions, confidence=0.95):
    """
    Calcula intervalo de confiança para previsões.
    
    Args:
        predictions: Array de probabilidades
        confidence: Nível de confiança (0.95 = 95%)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    mean = np.mean(predictions)
    std = np.std(predictions)
    n = len(predictions)
    
    # t-distribution para amostras pequenas
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * (std / np.sqrt(n))
    
    return (mean - margin, mean + margin)

def get_prediction_confidence(prob, spread=None):
    """
    Classifica confiança da previsão.
    
    Args:
        prob: Probabilidade do favorito (50-100)
        spread: Spread previsto (opcional)
    
    Returns:
        str: 'HIGH', 'MEDIUM', 'LOW'
    """
    # Baseado em quão próximo está de 50/50
    prob_diff = abs(prob - 50)
    
    if spread is not None:
        # Se temos spread, considerar também
        spread_abs = abs(spread)
        
        if prob_diff > 15 and spread_abs > 7:
            return 'HIGH'
        elif prob_diff > 8 and spread_abs > 4:
            return 'MEDIUM'
        else:
            return 'LOW'
    else:
        # Só probabilidade
        if prob_diff > 15:
            return 'HIGH'
        elif prob_diff > 5:
            return 'MEDIUM'
        else:
            return 'LOW'

def add_confidence_to_predictions(predictions_df):
    """
    Adiciona confiança às previsões.
    
    Args:
        predictions_df: DataFrame com colunas 'Prob Casa %', 'Spread Previsto'
    
    Returns:
        DataFrame com coluna 'Confidence' adicionada
    """
    confidences = []
    
    for _, row in predictions_df.iterrows():
        prob = max(row.get('Prob Casa %', 50), row.get('Prob Visitante %', 50))
        spread = row.get('Spread Previsto', None)
        
        confidence = get_prediction_confidence(prob, spread)
        confidences.append(confidence)
    
    predictions_df['Confidence'] = confidences
    
    return predictions_df

if __name__ == "__main__":
    # Test
    print(get_prediction_confidence(65, -8))  # HIGH
    print(get_prediction_confidence(56, -3))  # MEDIUM
    print(get_prediction_confidence(52, -1))  # LOW
