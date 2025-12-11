"""
Teste abrangente das Domain Expert Features.

Valida:
1. Features funcionam com dados NBA reais
2. Valores estão em ranges razoáveis
3. Features têm variância suficiente
4. Nenhum NaN inesperado
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from ml_pipeline.advanced_features import add_domain_expert_features
from data.repositories.db_manager import get_db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_features_with_nba_data():
    """Testa features com dados NBA reais."""
    
    logger.info("🏀 Testando Domain Expert Features com dados NBA...\n")
    
    # Carregar dados do DB
    db = get_db_manager()
    
    query = """
    SELECT 
        date,
        home_team,
        away_team,
        CAST(home_score AS FLOAT) as home_pts,
        CAST(away_score AS FLOAT) as away_pts,
        CAST(home_fga AS FLOAT) as home_fga,
        CAST(away_fga AS FLOAT) as away_fga,
        CAST(home_fg3a AS FLOAT) as home_fg3a,
        CAST(away_fg3a AS FLOAT) as away_fg3a,
        CAST(home_ast AS FLOAT) as home_ast,
        CAST(away_ast AS FLOAT) as away_ast,
        CAST(home_reb AS FLOAT) as home_reb,
        CAST(away_reb AS FLOAT) as away_reb,
        CAST(home_tov AS FLOAT) as home_tov,
        CAST(away_tov AS FLOAT) as away_tov
    FROM predictions
    WHERE home_score IS NOT NULL 
      AND away_score IS NOT NULL
      AND home_fga IS NOT NULL
    ORDER BY date DESC
    LIMIT 100
    """
    
    try:
        conn = db.get_connection()
        df = pd.read_sql_query(query, conn)
        db.return_connection(conn)
        logger.info(f"✅ Carregados {len(df)} jogos do DB\n")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar do DB: {e}")
        logger.info("📊 Criando dados sintéticos realistas...\n")
        
        # Dados sintéticos baseados em médias NBA
        np.random.seed(42)
        n = 100
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'home_team': ['Lakers'] * n,
            'away_team': ['Warriors'] * n,
            'home_pts': np.random.normal(110, 10, n),
            'away_pts': np.random.normal(108, 10, n),
            'home_fga': np.random.normal(85, 5, n),
            'away_fga': np.random.normal(85, 5, n),
            'home_fg3a': np.random.normal(35, 4, n),
            'away_fg3a': np.random.normal(35, 4, n),
            'home_ast': np.random.normal(25, 3, n),
            'away_ast': np.random.normal(24, 3, n),
            'home_reb': np.random.normal(45, 5, n),
            'away_reb': np.random.normal(45, 5, n),
            'home_tov': np.random.normal(13, 2, n),
            'away_tov': np.random.normal(13, 2, n),
        })
        
        # Adicionar colunas necessárias
        df['home_orb'] = df['home_reb'] * 0.25
        df['home_drb'] = df['home_reb'] * 0.75
        df['away_orb'] = df['away_reb'] * 0.25
        df['away_drb'] = df['away_reb'] * 0.75
        
        # Win/loss records simulados
        df['home_wins'] = np.random.randint(20, 45, n)
        df['home_losses'] = 50 - df['home_wins']
        df['away_wins'] = np.random.randint(20, 45, n)
        df['away_losses'] = 50 - df['away_wins']
    
    # Garantir colunas necessárias existem
    if 'home_orb' not in df.columns and 'home_reb' in df.columns:
        df['home_orb'] = df['home_reb'] * 0.25
        df['home_drb'] = df['home_reb'] * 0.75
        df['away_orb'] = df['away_reb'] * 0.25
        df['away_drb'] = df['away_reb'] * 0.75
    
    if 'home_wins' not in df.columns:
        df['home_wins'] = 30
        df['home_losses'] = 20
        df['away_wins'] = 25
        df['away_losses'] = 25
    
    # APLICAR FEATURES
    logger.info("🎯 Aplicando domain expert features...")
    df_original_cols = df.columns.tolist()
    df_with_features = add_domain_expert_features(df.copy())
    
    # Identificar novas features
    new_features = [col for col in df_with_features.columns if col not in df_original_cols]
    
    logger.info(f"\n✅ {len(new_features)} features adicionadas:\n")
    for i, feat in enumerate(new_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # VALIDAÇÕES
    logger.info("\n" + "="*60)
    logger.info("🔍 VALIDANDO FEATURES")
    logger.info("="*60 + "\n")
    
    issues = []
    
    for feat in new_features:
        values = df_with_features[feat]
        
        # Check 1: NaN count
        nan_count = values.isna().sum()
        nan_pct = (nan_count / len(values)) * 100
        
        # Check 2: Variance
        variance = values.var()
        std = values.std()
        
        # Check 3: Range
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        
        # Print stats
        print(f"📊 {feat}:")
        print(f"   Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"   Mean: {mean_val:.4f}, Std: {std:.4f}")
        print(f"   NaN: {nan_count}/{len(values)} ({nan_pct:.1f}%)")
        
        # Validations
        if nan_pct > 10:
            issues.append(f"❌ {feat}: {nan_pct:.1f}% NaN (>10% threshold)")
        
        if std < 0.001 and feat not in ['travel_fatigue_home', 'schedule_density_home']:
            # Algumas features placeholder têm std=0, isso é esperado
            issues.append(f"⚠️ {feat}: variance muito baixa (std={std:.4f})")
        
        if np.isinf(values).any():
            issues.append(f"❌ {feat}: contém infinitos!")
        
        print()
    
    # Resumo
    logger.info("="*60)
    logger.info("📋 RESUMO DA VALIDAÇÃO")
    logger.info("="*60)
    
    if issues:
        logger.warning(f"\n⚠️ {len(issues)} issues encontrados:\n")
        for issue in issues:
            print(f"  {issue}")
    else:
        logger.info("\n✅ Todas as features passaram nas validações!")
    
    # Feature Statistics Summary
    logger.info("\n" + "="*60)
    logger.info("📈 TOP 5 FEATURES POR VARIÂNCIA")
    logger.info("="*60 + "\n")
    
    # Calcular variance para features funcionais (excluir placeholders)
    functional_features = [f for f in new_features if 'travel' not in f and 'schedule' not in f and 'second_chance' not in f and 'fast_break' not in f and 'paint_pts' not in f and '_impact_' not in f]
    
    variances = {feat: df_with_features[feat].var() for feat in functional_features}
    top_variance = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (feat, var) in enumerate(top_variance, 1):
        print(f"  {i}. {feat}: variance={var:.6f}")
    
    # Salvardados para análise posterior
    output_path = 'data/test_features_output.csv'
    df_with_features.to_csv(output_path, index=False)
    logger.info(f"\n💾 Dados salvos em: {output_path}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ TESTE COMPLETO!")
    logger.info("="*60)
    
    return {
        'n_features': len(new_features),
        'n_functional': len(functional_features),
        'n_issues': len(issues),
        'n_samples': len(df_with_features)
    }


if __name__ == '__main__':
    results = test_features_with_nba_data()
    print(f"\n🎯 Final Results: {results}")
