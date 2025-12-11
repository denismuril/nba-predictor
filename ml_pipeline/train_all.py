#!/usr/bin/env python3
"""
Script unificado para treinar todos os modelos do NBA Predictor.

Executa em ordem:
1. Totals Model V18
2. Ensemble V6 (Moneyline)
3. Calibrador (Isotonic Regression)

Usage:
    python ml_pipeline/train_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

# Garantir que estamos no diretório correto
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def run_training(script_name: str, description: str) -> bool:
    """Executa um script de treinamento."""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    
    script_path = ROOT_DIR / "ml_pipeline" / script_name
    
    if not script_path.exists():
        print(f"❌ Script não encontrado: {script_path}")
        return False
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT_DIR)
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} concluído em {elapsed:.1f}s")
        return True
    else:
        print(f"❌ {description} falhou (código: {result.returncode})")
        return False


def main():
    print("\n" + "=" * 60)
    print("🏀 NBA PREDICTOR - TREINAMENTO COMPLETO")
    print("=" * 60)
    
    total_start = time.time()
    
    # Lista de scripts para treinar
    scripts = [
        ("train_totals_model.py", "Treinando Totals Model V18"),
        ("train_ensemble_v6.py", "Treinando Ensemble V6 (Moneyline)"),
        ("train_calibrator.py", "Treinando Calibrador"),
    ]
    
    results = []
    
    for script, description in scripts:
        success = run_training(script, description)
        results.append((description, success))
    
    # Resumo final
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("📊 RESUMO DO TREINAMENTO")
    print("=" * 60)
    
    all_success = True
    for description, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {description}")
        if not success:
            all_success = False
    
    print(f"\n⏱️  Tempo total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    if all_success:
        print("\n🎉 TODOS OS MODELOS TREINADOS COM SUCESSO!")
        print("\n📌 Próximo passo: python main.py --ml")
    else:
        print("\n⚠️  Alguns modelos falharam. Verifique os logs acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
