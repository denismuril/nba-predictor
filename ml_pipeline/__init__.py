"""
NBA Predictor ML Pipeline - v21.5 (Auditoria Forense Completa)
===============================================================

Módulos consolidados após auditoria de código v21.4.

Uso recomendado (imports diretos):
    from ml_pipeline.train_ensemble_v6 import train_ensemble_model_v6
    from ml_pipeline.data_preparation import load_historical_data
"""

__version__ = "21.5"

# Lazy imports para evitar dependências circulares
# Os módulos são importados apenas quando acessados

def __getattr__(name):
    """Lazy import para módulos principais."""
    if name == 'train_model':
        from ml_pipeline.train_ensemble_v6 import train_ensemble_model_v6
        return train_ensemble_model_v6
    elif name == 'load_historical_data':
        from ml_pipeline.data_preparation import load_historical_data
        return load_historical_data
    elif name == 'prepare_data_for_training':
        from ml_pipeline.data_preparation import prepare_data_for_training
        return prepare_data_for_training
    elif name == 'NBAEloSystem':
        from ml_pipeline.elo_system import NBAEloSystem
        return NBAEloSystem
    raise AttributeError(f"module 'ml_pipeline' has no attribute '{name}'")
