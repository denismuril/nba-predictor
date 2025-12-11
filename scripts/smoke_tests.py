#!/usr/bin/env python3
"""
Smoke Tests - Testes de Integração End-to-End
Valida que todo o sistema funciona corretamente após mudanças
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmokeTestSuite:
    """Suite de testes end-to-end"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
    
    def test(self, name, func):
        """Execute um teste e registre resultado"""
        try:
            logger.info(f"\n🧪 Testando: {name}...")
            func()
            self.tests_passed += 1
            logger.info(f"   ✅ PASSOU")
        except Exception as e:
            self.tests_failed += 1
            self.errors.append((name, str(e)))
            logger.error(f"   ❌ FALHOU: {e}")
    
    def report(self):
        """Gera relatório final"""
        logger.info("\n" + "="*80)
        logger.info("📊 RELATÓRIO DE TESTES")
        logger.info("="*80)
        logger.info(f"✅ Passaram: {self.tests_passed}")
        logger.info(f"❌ Falharam: {self.tests_failed}")
        logger.info(f"📈 Taxa de sucesso: {self.tests_passed/(self.tests_passed+self.tests_failed)*100:.1f}%")
        
        if self.errors:
            logger.error("\n⚠️  ERROS ENCONTRADOS:")
            for name, error in self.errors:
                logger.error(f"   - {name}: {error}")
        
        return self.tests_failed == 0

# Inicializar suite
suite = SmokeTestSuite()

logger.info("="*80)
logger.info("🚀 SMOKE TESTS - INTEGRAÇÃO END-TO-END")
logger.info("="*80)

# ============================================================================
# TESTE 1: Importações Críticas
# ============================================================================
def test_imports():
    """Testa se todos os módulos críticos podem ser importados"""
    from ml_pipeline.data_preparation import load_historical_data
    from data.scrapers.odds_scraper import obter_odds
    from data.repositories.db_manager import get_db_manager
    from exceptions.odds_exceptions import OddsUnavailableError
    logger.info("   Todas as importações funcionaram")

suite.test("Importações Críticas", test_imports)

# ============================================================================
# TESTE 2: Conexão com Banco de Dados
# ============================================================================
def test_database():
    """Testa conexão com PostgreSQL"""
    from data.repositories.db_manager import get_db_manager
    import pandas as pd
    db = get_db_manager()
    conn = db.get_connection()
    df = pd.read_sql("SELECT 1 as test", conn)
    assert df is not None and len(df) > 0, "Query falhou"
    conn.close()
    logger.info(f"   DB conectado: {db.db_type}")

suite.test("Conexão com Banco de Dados", test_database)

# ============================================================================
# TESTE 3: Carregamento de Dados Históricos
# ============================================================================
def test_data_loading():
    """Testa carregamento de dados com features corrigidas"""
    from ml_pipeline.data_preparation import load_historical_data
    df, weights = load_historical_data(
        seasons=['2024-25', '2025-26'],
        apply_weights=True
    )
    assert df is not None, "DataFrame vazio"
    assert len(df) > 0, "Sem dados"
    assert 'home_rest_days' in df.columns, "home_rest_days missing"
    assert 'away_rest_days' in df.columns, "away_rest_days missing"
    
    # Verificar que rest_days está limitado a 7
    max_home_rest = df['home_rest_days'].max()
    max_away_rest = df['away_rest_days'].max()
    assert max_home_rest <= 7, f"home_rest_days não limitado! Max = {max_home_rest}"
    assert max_away_rest <= 7, f"away_rest_days não limitado! Max = {max_away_rest}"
    
    logger.info(f"   {len(df)} jogos carregados, rest_days ≤ 7 ✓")

suite.test("Carregamento de Dados Históricos", test_data_loading)

# ============================================================================
# TESTE 4: Modelo Existe e Pode ser Carregado
# ============================================================================
def test_model_loading():
    """Testa se modelo pode ser carregado"""
    import joblib
    import os
    
    model_path = 'models/ensemble_v7.joblib'
    assert os.path.exists(model_path), f"Modelo não encontrado: {model_path}"
    
    model = joblib.load(model_path)
    assert model is not None, "Modelo vazio"
    
    logger.info(f"   Modelo carregado: {type(model).__name__}")

suite.test("Modelo Existe e Carrega", test_model_loading)

# ============================================================================
# TESTE 5: Features Não Têm Leakage
# ============================================================================
def test_no_leakage():
    """Verifica que features críticas de leakage não estão presentes"""
    import joblib
    
    features = joblib.load('models/feature_names_v7.joblib')
    
    # Features proibidas (causam leakage)
    forbidden = ['winner', 'correct', 'home_score', 'away_score', 
                 'total_points', 'pt_diff']
    
    for feat in forbidden:
        assert feat not in features, f"LEAKAGE DETECTADO: {feat} nas features!"
    
    logger.info(f"   {len(features)} features, nenhuma com leakage ✓")

suite.test("Features Não Têm Leakage", test_no_leakage)

# ============================================================================
# TESTE 6: Sistema de Odds (Sem Mock Fallback)
# ============================================================================
def test_odds_no_mock():
    """Testa que sistema de odds não usa fallback para mock"""
    from data.scrapers.odds_scraper import obter_odds
    from exceptions.odds_exceptions import OddsUnavailableError
    
    # Se nenhuma API key configurada, deve lançar erro (não usar mock)
    import os
    old_keys = {
        'ODDS_API_KEY': os.environ.get('ODDS_API_KEY'),
        'SPORTSDATA_API_KEY': os.environ.get('SPORTSDATA_API_KEY'),
        'RAPIDAPI_KEY': os.environ.get('RAPIDAPI_KEY')
    }
    
    # Remover temporariamente
    for key in old_keys:
        if key in os.environ:
            del os.environ[key]
    
    try:
        odds = obter_odds()
        # Se chegou aqui, deveria ter lançado erro!
        raise AssertionError("Sistema usou mock ao invés de lançar erro!")
    except OddsUnavailableError:
        logger.info("   Sistema corretamente lança erro sem API keys ✓")
    finally:
        # Restaurar keys
        for key, value in old_keys.items():
            if value:
                os.environ[key] = value

suite.test("Sistema de Odds (Sem Mock)", test_odds_no_mock)

# ============================================================================
# TESTE 7: Exception Handling Funciona
# ============================================================================
def test_exceptions():
    """Testa que exceptions customizadas funcionam"""
    from exceptions.odds_exceptions import OddsUnavailableError, OddsAPIKeyMissingError
    
    try:
        raise OddsUnavailableError("Teste")
    except OddsUnavailableError as e:
        assert "Teste" in str(e)
    
    try:
        raise OddsAPIKeyMissingError("TestAPI")
    except OddsAPIKeyMissingError as e:
        assert "TestAPI" in str(e)
    
    logger.info("   Exceptions funcionando corretamente ✓")

suite.test("Exception Handling", test_exceptions)

# ============================================================================
# TESTE 8: .env.example Existe
# ============================================================================
def test_env_example():
    """Verifica que template de .env existe"""
    import os
    assert os.path.exists('.env.example'), ".env.example não encontrado!"
    
    with open('.env.example', 'r') as f:
        content = f.read()
        assert 'ODDS_API_KEY' in content, "ODDS_API_KEY missing"
        assert 'SPORTSDATA_API_KEY' in content, "SPORTSDATA_API_KEY missing"
        assert 'RAPIDAPI_KEY' in content, "RAPIDAPI_KEY missing"
    
    logger.info("   .env.example completo ✓")

suite.test(".env.example Existe", test_env_example)

# ============================================================================
# RELATÓRIO FINAL
# ============================================================================
success = suite.report()

if success:
    logger.info("\n🎉 TODOS OS TESTES PASSARAM! Sistema pronto para produção.")
    sys.exit(0)
else:
    logger.error("\n❌ ALGUNS TESTES FALHARAM! Revise os erros acima.")
    sys.exit(1)
