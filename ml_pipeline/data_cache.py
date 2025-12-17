#!/usr/bin/env python3
"""
Data Cache V22.0 - Acelera carregamento de dados preparados.

O load_historical_data demora ~12-15 horas porque recalcula:
- Rolling features (5, 10, 30 jogos)
- Referee stats (expanding window)
- Travel fatigue (distância + descanso)
- Player impact (RAPM)
- etc.

Este cache salva o DataFrame preparado e reutiliza se < 24h.
"""
import pandas as pd
import joblib
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DIR = 'data/cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'prepared_data_cache.joblib')
CACHE_METADATA = os.path.join(CACHE_DIR, 'cache_metadata.joblib')
CACHE_MAX_AGE_HOURS = 24


def ensure_cache_dir():
    """Garante que o diretório de cache existe."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_age_hours():
    """Retorna a idade do cache em horas, ou None se não existir."""
    if not os.path.exists(CACHE_METADATA):
        return None
    try:
        metadata = joblib.load(CACHE_METADATA)
        created = metadata.get('created_at')
        if created:
            age = (datetime.now() - created).total_seconds() / 3600
            return age
    except Exception:
        pass
    return None


def is_cache_valid():
    """Verifica se o cache existe e é recente o suficiente."""
    age = get_cache_age_hours()
    if age is None:
        return False
    return age < CACHE_MAX_AGE_HOURS


def load_cached_data():
    """Carrega dados do cache se válido."""
    if not is_cache_valid():
        logger.info("❌ Cache inválido ou expirado")
        return None
    
    try:
        logger.info(f"📦 Carregando dados do cache (idade: {get_cache_age_hours():.1f}h)...")
        df = joblib.load(CACHE_FILE)
        logger.info(f"   ✅ Cache carregado: {len(df)} jogos")
        return df
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar cache: {e}")
        return None


def save_to_cache(df):
    """Salva DataFrame preparado no cache."""
    try:
        ensure_cache_dir()
        logger.info(f"💾 Salvando {len(df)} jogos no cache...")
        joblib.dump(df, CACHE_FILE)
        joblib.dump({
            'created_at': datetime.now(),
            'num_games': len(df),
            'num_features': len(df.columns)
        }, CACHE_METADATA)
        logger.info("   ✅ Cache salvo com sucesso!")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao salvar cache: {e}")


def invalidate_cache():
    """Remove o cache para forçar recálculo."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    if os.path.exists(CACHE_METADATA):
        os.remove(CACHE_METADATA)
    logger.info("🗑️ Cache invalidado")


def load_historical_data_cached(seasons=None, force_refresh=False, **kwargs):
    """
    Wrapper para load_historical_data com cache.
    
    Args:
        seasons: Temporadas a carregar
        force_refresh: Se True, ignora cache e recalcula
        **kwargs: Argumentos para load_historical_data
    
    Returns:
        DataFrame preparado (do cache ou recalculado)
    """
    from ml_pipeline.data_preparation import load_historical_data
    
    # Tentar cache primeiro
    if not force_refresh:
        cached = load_cached_data()
        if cached is not None:
            return cached
    
    # Calcular do zero (demorado)
    logger.info("🔄 Calculando dados do zero (isso pode demorar)...")
    df = load_historical_data(seasons=seasons, **kwargs)
    
    # Salvar no cache para próxima vez
    save_to_cache(df)
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Testar cache
    print("Verificando cache...")
    age = get_cache_age_hours()
    if age:
        print(f"Cache existe, idade: {age:.1f} horas")
    else:
        print("Cache não existe")
    
    print(f"Cache válido: {is_cache_valid()}")
