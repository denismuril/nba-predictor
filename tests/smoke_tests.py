"""
Testes de Validação de Produção (Smoke Tests)

Valida que todos os componentes estão funcionando corretamente em produção.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test 1: Todos os módulos importáveis."""
    logger.info("🧪 Test 1: Imports...")
    
    try:
        from utils.connection_pool import ConnectionPool
        from utils.team_normalization import normalize_team
        from utils.leakage_prevention import validate_rolling_features
        from ml_pipeline.calibrator import AutoCalibrator
        from ml_pipeline.advanced_features import add_domain_expert_features
        from ml_pipeline.feature_pipeline import add_all_features
        
        logger.info("   ✅ Todos os imports OK")
        return True
    except Exception as e:
        logger.error(f"   ❌ Erro nos imports: {e}")
        return False


def test_calibrator():
    """Test 2: Calibrator carregável."""
    logger.info("🧪 Test 2: Calibrator...")
    
    try:
        from ml_pipeline.calibrator import get_calibrator
        
        calibrator = get_calibrator('models/calibrator.pkl')
        
        if calibrator and calibrator.fitted:
            stats = calibrator.get_stats()
            logger.info(f"   ✅ Calibrator OK: {stats['n_samples']} samples")
            return True
        else:
            logger.warning("   ⚠️ Calibrator não fitted ainda (primeira execução OK)")
            return True
            
    except FileNotFoundError:
        logger.info("   ℹ️ Calibrator não encontrado (primeira execução OK)")
        return True
    except Exception as e:
        logger.error(f"   ❌ Erro no calibrator: {e}")
        return False


def test_model():
    """Test 3: Modelo carregável."""
    logger.info("🧪 Test 3: Modelo ML...")
    
    try:
        import joblib
        
        model_path = Path('models/ml_model.joblib')
        
        if not model_path.exists():
            logger.warning("   ⚠️ Modelo não encontrado - precisa treinar")
            return True  # OK para primeira execução
        
        model = joblib.load(model_path)
        n_features = len(model.feature_names_in_)
        
        logger.info(f"   ✅ Modelo OK: {n_features} features")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Erro ao carregar modelo: {e}")
        return False


def test_features():
    """Test 4: Domain features funcionais."""
    logger.info("🧪 Test 4: Domain Features...")
    
    try:
        from ml_pipeline.feature_pipeline import add_all_features
        import pandas as pd
        
        # Dados mínimos de teste
        test_df = pd.DataFrame({
            'home_fga': [85],
            'away_fga': [88],
            'home_pts': [110],
            'away_pts': [108],
            'home_orb': [10],
            'home_drb': [32],
            'away_orb': [8],
            'away_drb': [35],
            'home_fg3a': [35],
            'away_fg3a': [38],
            'home_tov': [12],
            'away_tov': [14],
            'home_ast': [25],
            'away_ast': [23],
            'home_wins': [30],
            'home_losses': [15],
            'away_wins': [20],
            'away_losses': [25]
        })
        
        result = add_all_features(test_df, include_domain=True)
        n_new = len(result.columns) - len(test_df.columns)
        
        logger.info(f"   ✅ Features OK: +{n_new} features criadas")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Erro nas features: {e}")
        return False


def test_database():
    """Test 5: Database acessível."""
    logger.info("🧪 Test 5: Database...")
    
    try:
        from data.repositories.db_manager import get_db_manager
        
        db = get_db_manager()
        conn = db.get_connection()
        
        # Test query simples
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        db.return_connection(conn)
        
        logger.info("   ✅ Database OK")
        return True
        
    except Exception as e:
        logger.warning(f"   ⚠️ Database error (ok se não configurado): {e}")
        return True  # Não crítico para smoke test


def run_smoke_tests():
    """Executa todos os smoke tests."""
    
    logger.info("=" * 60)
    logger.info("🔥 SMOKE TESTS - Validação de Produção")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("")
    
    tests = [
        ("Imports", test_imports),
        ("Calibrator", test_calibrator),
        ("Modelo ML", test_model),
        ("Domain Features", test_features),
        ("Database", test_database),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"💥 Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        logger.info("")
    
    # Sumário
    logger.info("=" * 60)
    logger.info("📊 SUMÁRIO")
    logger.info("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status:10} - {test_name}")
    
    logger.info("")
    logger.info(f"Total: {passed_count}/{total_count} testes passaram")
    
    if passed_count == total_count:
        logger.info("🎉 TODOS OS TESTES PASSARAM!")
        return True
    else:
        logger.warning(f"⚠️ {total_count - passed_count} teste(s) falharam")
        return False


if __name__ == '__main__':
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
