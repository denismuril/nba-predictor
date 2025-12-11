"""
Módulo de Incerteza e Intervalos de Confiança.

Implementa bootstrapping para estimar a incerteza das predições do modelo.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

def estimate_confidence_interval(model, X: pd.DataFrame, n_bootstraps: int = 50, alpha: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima intervalos de confiança para predições usando bootstrapping dos estimadores do modelo.
    
    Funciona melhor com Ensembles (RandomForest, Stacking) que possuem múltiplos estimadores.
    
    Args:
        model: O modelo treinado (deve ter atributo estimators_ ou similar)
        X: DataFrame com as features para predição
        n_bootstraps: Número de amostras bootstrap (se aplicável)
        alpha: Nível de confiança (ex: 0.95 para 95%)
        
    Returns:
        Tuple[lower_bound, upper_bound]: Arrays com limites inferior e superior das probabilidades
    """
    # Verificar se é um modelo compatível (Ensemble/Stacking)
    predictions = []
    
    # Caso 1: StackingClassifier (tem estimators_)
    if hasattr(model, 'estimators_'):
        # Coletar predições de cada modelo base
        for estimator in model.estimators_:
            try:
                # Alguns estimadores podem ser pipelines
                if hasattr(estimator, 'predict_proba'):
                    pred = estimator.predict_proba(X)[:, 1] # Probabilidade da classe positiva (HOME)
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Falha ao obter predição de estimador: {e}")
                
    # Caso 2: RandomForest/ExtraTrees (tem estimators_)
    elif hasattr(model, 'estimators_') and isinstance(model.estimators_, list):
        # Para RF, podemos usar uma amostra dos estimadores para ser mais rápido
        # Se tiver 200 árvores, usar todas pode ser lento. Vamos usar n_bootstraps.
        estimators_to_use = model.estimators_
        if len(estimators_to_use) > n_bootstraps:
            estimators_to_use = np.random.choice(estimators_to_use, n_bootstraps, replace=False)
            
        for estimator in estimators_to_use:
            try:
                pred = estimator.predict_proba(X)[:, 1]
                predictions.append(pred)
            except Exception as e:
                pass
                
    # Caso 3: Modelo único (não ensemble) ou não suportado
    else:
        logger.warning("Modelo não suporta bootstrapping nativo (não é ensemble exposto). Retornando predição padrão sem intervalo.")
        try:
            pred = model.predict_proba(X)[:, 1]
            return pred, pred # Intervalo zero
        except:
            return np.zeros(len(X)), np.zeros(len(X))
            
    if not predictions:
        logger.warning("Nenhuma predição coletada para bootstrapping.")
        return np.zeros(len(X)), np.zeros(len(X))
        
    # Converter para array numpy (n_estimators, n_samples)
    predictions = np.array(predictions)
    
    # Calcular percentis
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    lower_bound = np.percentile(predictions, lower_p, axis=0)
    upper_bound = np.percentile(predictions, upper_p, axis=0)
    
    return lower_bound, upper_bound

def format_confidence_interval(prob: float, lower: float, upper: float) -> str:
    """Formata o intervalo de confiança para exibição (ex: '60% ± 5%')"""
    # Margem de erro média
    margin = (upper - lower) / 2
    return f"{prob*100:.1f}% ± {margin*100:.1f}%"
