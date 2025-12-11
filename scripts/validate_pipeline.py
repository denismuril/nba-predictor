#!/usr/bin/env python
"""
Script de validação completa do pipeline de predição.

Testa:
1. Carregamento do modelo e calibrador
2. Geração de features
3. Previsões calibradas
4. Flags de confiança
5. Comparação antes/depois da calibração

Usage:
    python scripts/validate_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.predict import predict_next_games
from utils.logger_config import get_logger

logger = get_logger(__name__)


def validate_pipeline():
    """Executa validação completa do pipeline."""
    
    print("\n" + "="*60)
    print("🧪 VALIDAÇÃO COMPLETA DO PIPELINE DE PREDIÇÃO")
    print("="*60 + "\n")
    
    # 1. Testar predict_next_games
    print("📊 Testando predict_next_games()...")
    results = predict_next_games()
    
    if results.empty:
        print("⚠️  Nenhum jogo encontrado para hoje. Isso é esperado se não houver jogos agendados.")
        return
    
    # 2. Validar colunas esperadas
    expected_columns = [
        'home_team', 'away_team',
        'prob_home_raw', 'prob_ml_home', 'prob_ml_away',
        'confidence_score', 'confidence_level',
        'ci_lower', 'ci_upper'
    ]
    
    missing_cols = [col for col in expected_columns if col not in results.columns]
    if missing_cols:
        print(f"❌ Colunas ausentes: {missing_cols}")
        return
    
    print(f"✅ Todas as colunas esperadas presentes")
    
    # 3. Analisar resultados
    print(f"\n📈 RESULTADOS ({len(results)} jogos):\n")
    
    for idx, row in results.iterrows():
        print(f"🏀 {row['home_team']} vs {row['away_team']}")
        print(f"   Prob Raw:    {row['prob_home_raw']:.3f}")
        print(f"   Prob Calib:  {row['prob_ml_home']:.3f}")
        print(f"   Diferença:   {row['prob_ml_home'] - row['prob_home_raw']:+.3f}")
        print(f"   Confiança:   {row['confidence_level']} ({row['confidence_score']:.3f})")
        print(f"   IC 95%:      [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
        print()
    
    # 4. Estatísticas de calibração
    print("\n📊 ESTATÍSTICAS DE CALIBRAÇÃO:")
    print(f"   Média Raw:        {results['prob_home_raw'].mean():.3f}")
    print(f"   Média Calibrada:  {results['prob_ml_home'].mean():.3f}")
    print(f"   Ajuste médio:     {(results['prob_ml_home'] - results['prob_home_raw']).mean():+.3f}")
    print(f"   Std ajuste:       {(results['prob_ml_home'] - results['prob_home_raw']).std():.3f}")
    
    # 5. Distribuição de confiança
    print("\n📊 DISTRIBUIÇÃO DE CONFIANÇA:")
    conf_dist = results['confidence_level'].value_counts()
    for level, count in conf_dist.items():
        pct = count / len(results) * 100
        print(f"   {level}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO")
    print("="*60 + "\n")


if __name__ == '__main__':
    try:
        validate_pipeline()
    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
