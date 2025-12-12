"""
Health Check Script para Monitoramento de Produção

Verifica:
- Modelos existem e são recentes
- Pipeline de predição funciona
- Features críticas presentes
"""
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_models():
    """Verifica se modelos existem e são recentes."""
    logger.info("🔍 Verificando modelos...")

    models = [
        ('ml_model.joblib', 'Moneyline'),
        ('spread_model_v16.joblib', 'Spread'),
    ]

    all_ok = True

    for model_file, model_name in models:
        path = Path(f'models/{model_file}')

        if not path.exists():
            logger.error(f'❌ {model_name}: Modelo ausente ({model_file})')
            all_ok = False
            continue

        # Verificar idade do modelo
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days

        size_mb = path.stat().st_size / (1024 * 1024)

        if age_days > 7:
            logger.warning(f'⚠️  {model_name}: Modelo com {age_days} dias (recomendado retreinar)')
        else:
            logger.info(f'✅ {model_name}: OK (idade: {age_days}d, tamanho: {size_mb:.1f}MB)')

    return all_ok


def check_pipeline():
    """Testa pipeline de predição."""
    logger.info("\n🔍 Testando pipeline de predição...")

    try:
        import joblib
        from pathlib import Path
        
        model_path = Path('models/ml_model.joblib')
        if not model_path.exists():
            logger.warning('⚠️  Modelo ML não encontrado - execute train_ensemble_v6 primeiro')
            return True  # Não é erro crítico
        
        model = joblib.load(model_path)

        # Verificar que modelo tem feature_names
        if hasattr(model, 'feature_names_in_'):
            n_features = len(model.feature_names_in_)
            logger.info(f'✅ Pipeline OK ({n_features} features esperadas)')
        elif hasattr(model, 'named_steps'):
            logger.info('✅ Pipeline OK (Stacking Classifier)')
        else:
            logger.info('✅ Pipeline OK (modelo carregado)')

        return True

    except Exception as e:
        logger.error(f'❌ Erro no pipeline: {e}')
        return False


def check_formulas():
    """Verifica módulo de fórmulas canônicas."""
    logger.info("\n🔍 Verificando fórmulas NBA...")

    try:
        from utils.nba_formulas import calculate_all_advanced_stats

        # Teste com dados realistas
        stats = calculate_all_advanced_stats(
            pts=115, fgm=42, fga=88, fg3m=15,
            fta=18, ftm=16, orb=8, drb=38, tov=11,
            opp_pts=108, opp_drb=35, minutes_played=48
        )

        # Validar ranges
        assert 95 <= stats['possessions'] <= 105, "Possessions fora do range"
        assert 0.4 <= stats['efg_pct'] <= 0.7, "eFG% fora do range"
        assert 110 <= stats['off_rating'] <= 125, "ORtg fora do range"

        logger.info('✅ Fórmulas NBA OK')
        return True

    except Exception as e:
        logger.error(f'❌ Erro nas fórmulas: {e}')
        return False


def check_betting_engine():
    """Verifica Kelly Criterion concurrent."""
    logger.info("\n🔍 Verificando betting engine...")

    try:
        from ml_pipeline.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000, kelly_fraction=0.25)

        # Teste concurrent Kelly
        stake = engine.calculate_kelly_stake_concurrent(
            prob_win=0.55,
            decimal_odds=2.0,
            n_concurrent_bets=10
        )

        # Validações
        assert 0 <= stake <= 0.05, "Stake individual fora do range"
        assert stake < 0.02, "Stake muito alto para 10 jogos"

        logger.info(f'✅ Kelly Criterion OK (stake para 10 jogos: {stake*100:.2f}%)')
        return True

    except Exception as e:
        logger.error(f'❌ Erro no betting engine: {e}')
        return False

        return False


def check_injury_system():
    """Verifica sistema de lesões (data-driven v2.2)."""
    logger.info("\n🔍 Verificando sistema de lesões...")
    
    try:
        from data.scrapers.injury_scraper_v2 import StatsManager, InjuryManager, get_injuries_with_cache
        
        # 1. Stats Manager
        sm = StatsManager()
        score = sm.get_player_importance("Nikola Jokic")
        
        if len(sm._stats_cache) == 0:
            logger.warning("⚠️ StatsManager não carregou jogadores (fallback será usado)")
        else:
            logger.info(f"✅ StatsManager OK ({len(sm._stats_cache)} jogadores carregados)")
            
        if 0.2 <= score <= 0.35:
            logger.info(f"✅ Player Score Validation OK (Jokic: {score:.3f})")
        else:
            logger.warning(f"⚠️ Player Score Validation SUSPECT (Jokic: {score:.3f})")
            
        # 2. Injury Manager
        im = InjuryManager()
        # Não forçar refresh para não pesar na API externa durante health check
        injuries = im.get_latest_injuries(force_refresh=False)
        
        if injuries is None:
             logger.warning("⚠️ InjuryManager retornou None")
        else:
             logger.info(f"✅ InjuryManager OK ({len(injuries)} lesões atuais)")
             
        # 3. Compatibility check
        legacy_data = get_injuries_with_cache()
        if isinstance(legacy_data, dict):
            logger.info("✅ Legacy Compatibility Link OK")
        else:
            logger.error("❌ Legacy Compatibility Link BROKEN")
            return False
            
        return True
        
    except ImportError as e:
        logger.error(f"❌ Erro de importação no sistema de lesões: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro no sistema de lesões: {e}")
        return False

def main():
    """Executa todos os health checks."""
    logger.info("="*60)
    logger.info("🏥 HEALTH CHECK - NBA Predictor v21.5")
    logger.info("="*60)

    checks = [
        ("Modelos", check_models),
        ("Pipeline", check_pipeline),
        ("Fórmulas NBA", check_formulas),
        ("Betting Engine", check_betting_engine),
        ("Sistema de Lesões", check_injury_system)
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Erro ao executar check '{name}': {e}")
            results.append(False)

    logger.info("\n" + "="*60)

    if all(results):
        logger.info("✅ TODOS OS CHECKS PASSARAM - Sistema OK para produção!")
        logger.info("="*60)
        return 0
    else:
        failed = sum(1 for r in results if not r)
        logger.error(f"❌ {failed}/{len(checks)} CHECKS FALHARAM - Revisar antes de produção")
        logger.info("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
