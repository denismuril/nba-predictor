"""
Backtesting de Domain Expert Features

Valida importância e impacto das features via ablation study.

Usage:
    python tests/backtest_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBacktest:
    """Backtest framework para Domain Expert Features."""
    
    def __init__(self):
        self.results = {}
    
    def load_data_with_features(self):
        """Carrega dados com domain features."""
        
        logger.info("📂 Carregando dados com domain features...")
        
        try:
            from ml_pipeline.data_preparation import load_multi_season_data
            from ml_pipeline.feature_pipeline import add_all_features
            
            df = load_multi_season_data(seasons=['2024-25'])
            
            if df.empty:
                raise ValueError("No data")
            
            # Add features
            df = add_all_features(df, include_domain=True)
            
            logger.info(f"✅ {len(df)} jogos, {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Erro: {e}. Usando dados sintéticos...")
            
            # Dados sintéticos
            np.random.seed(42)
            n = 500
            
            from ml_pipeline.feature_pipeline import add_all_features
            
            df = pd.DataFrame({
                'home_fga': np.random.normal(85, 5, n),
                'away_fga': np.random.normal(85, 5, n),
                'home_pts': np.random.normal(110, 10, n),
                'away_pts': np.random.normal(108, 10, n),
                'home_orb': np.random.normal(11, 2, n),
                'home_drb': np.random.normal(34, 3, n),
                'away_orb': np.random.normal(11, 2, n),
                'away_drb': np.random.normal(34, 3, n),
                'home_fg3a': np.random.normal(35, 4, n),
                'away_fg3a': np.random.normal(35, 4, n),
                'home_tov': np.random.normal(13, 2, n),
                'away_tov': np.random.normal(13, 2, n),
                'home_ast': np.random.normal(25, 3, n),
                'away_ast': np.random.normal(24, 3, n),
                'home_wins': np.random.randint(20, 45, n),
                'home_losses': 50,
                'away_wins': np.random.randint(20, 45, n),
                'away_losses': 50
            })
            
            df['home_losses'] = 50 - df['home_wins']
            df['away_losses'] = 50 - df['away_wins']
            
            # Add domain features
            df = add_all_features(df, include_domain=True)
            
            # Target
            df['target'] = (df['home_pts'] > df['away_pts']).astype(int)
            
            return df
    
    def identify_domain_features(self, df: pd.DataFrame) -> list:
        """Identifica domain expert features no DataFrame."""
        
        domain_keywords = [
            'pace', 'def_matchup', 'reb_edge', '3pt', 'tov_pressure',
            'clutch', 'playoff', 'ts_pct', 'ast_tov'
        ]
        
        domain_features = [
            col for col in df.columns
            if any(kw in col.lower() for kw in domain_keywords)
        ]
        
        logger.info(f"🎯 {len(domain_features)} domain features identificadas")
        return domain_features
    
    def ablation_study(self, df: pd.DataFrame):
        """
        Ablation study: treina modelo com/sem domain features.
        
        Returns:
            Dict com resultados comparativos
        """
        logger.info("🔬 Iniciando Ablation Study...")
        
        # Preparar features
        feature_cols = [col for col in df.columns if col not in [
            'date', 'home_team', 'away_team', 'home_pts', 'away_pts',
            'target', 'home_score', 'away_score', 'home_losses', 'away_losses'
        ]]
        
        domain_features = self.identify_domain_features(df)
        base_features = [f for f in feature_cols if f not in domain_features]
        
        # Remove NaN
        df_clean = df[feature_cols + ['target']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # Baseline: Sem domain features
        logger.info("\n1️⃣ Baseline (sem domain features):")
        X_train_base = X_train[base_features]
        X_test_base = X_test[base_features]
        
        model_base = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model_base.fit(X_train_base, y_train)
        
        y_pred_base = model_base.predict(X_test_base)
        y_proba_base = model_base.predict_proba(X_test_base)[:, 1]
        
        results['baseline'] = {
            'accuracy': accuracy_score(y_test, y_pred_base),
            'auc': roc_auc_score(y_test, y_proba_base),
            'n_features': len(base_features)
        }
        
        logger.info(f"   Accuracy: {results['baseline']['accuracy']:.4f}")
        logger.info(f"   AUC: {results['baseline']['auc']:.4f}")
        logger.info(f"   Features: {results['baseline']['n_features']}")
        
        # Full model: Com domain features
        logger.info("\n2️⃣ Full Model (com domain features):")
        
        model_full = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model_full.fit(X_train, y_train)
        
        y_pred_full = model_full.predict(X_test)
        y_proba_full = model_full.predict_proba(X_test)[:, 1]
        
        results['full'] = {
            'accuracy': accuracy_score(y_test, y_pred_full),
            'auc': roc_auc_score(y_test, y_proba_full),
            'n_features': len(feature_cols)
        }
        
        logger.info(f"   Accuracy: {results['full']['accuracy']:.4f}")
        logger.info(f"   AUC: {results['full']['auc']:.4f}")
        logger.info(f"   Features: {results['full']['n_features']}")
        
        # Delta
        results['improvement'] = {
            'accuracy_delta': (results['full']['accuracy'] - results['baseline']['accuracy']) * 100,
            'auc_delta': (results['full']['auc'] - results['baseline']['auc']) * 100
        }
        
        logger.info("\n📊 Melhoria com Domain Features:")
        logger.info(f"   Accuracy: +{results['improvement']['accuracy_delta']:.2f}%")
        logger.info(f"   AUC: +{results['improvement']['auc_delta']:.2f}%")
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': model_full.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Top domain features
        domain_importances = importances[
            importances['feature'].isin(domain_features)
        ].head(10)
        
        logger.info("\n🎯 Top 10 Domain Features:")
        for idx, row in domain_importances.iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        results['feature_importance'] = importances.to_dict('records')
        results['top_domain_features'] = domain_importances.to_dict('records')
        
        return results
    
    def plot_feature_importance(self, results: dict, save_path: str = 'reports/feature_importance.png'):
        """Plota feature importance."""
        
        importances = pd.DataFrame(results['feature_importance'])
        top_20 = importances.head(20)
        
        # Identif domain vs base
        domain_features = [f['feature'] for f in results['top_domain_features']]
        colors = ['green' if f in domain_features else 'blue' for f in top_20['feature']]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(top_20))
        ax.barh(y_pos, top_20['importance'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_20['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Feature Importance (Verde = Domain Features)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        logger.info(f"\n📊 Gráfico salvo em: {save_path}")
        plt.close()


def run_feature_backtest():
    """Executa backtesting de features."""
    
    logger.info("="*60)
    logger.info("🔬 BACKTESTING: Domain Expert Features")
    logger.info("="*60 + "\n")
    
    backtester = FeatureBacktest()
    
    # Load data
    df = backtester.load_data_with_features()
    
    # Ablation study
    results = backtester.ablation_study(df)
    
    # Plot
    backtester.plot_feature_importance(results)
    
    # Save report
    report_path = Path('reports/feature_backtest_report.json')
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Relatório salvo em: {report_path}")
    logger.info("\n" + "="*60)
    logger.info("✅ BACKTESTING COMPLETO!")
    logger.info("="*60)
    
    return results


if __name__ == '__main__':
    results = run_feature_backtest()
    
    print(f"\n📊 Resumo:")
    print(f"   Baseline accuracy: {results['baseline']['accuracy']:.2%}")
    print(f"   Full model accuracy: {results['full']['accuracy']:.2%}")
    print(f"   Melhoria: +{results['improvement']['accuracy_delta']:.2f}%")
