#!/usr/bin/env python3
"""
Script para limpar modelos obsoletos.
Mantém apenas os modelos essenciais em produção.
"""
import os
from pathlib import Path
import shutil

MODELS_DIR = Path("data/models")

# Modelos ESSENCIAIS (não remover)
KEEP = {
    # Ensemble V6 (Moneyline)
    "ensemble_model_v6.joblib",
    "feature_names_v6.joblib",
    
    # Totals V18
    "totals_model_v18.joblib",
    "totals_feature_names_v18.joblib",
    
    # Calibrador
    "calibrator.pkl",
    
    # Elo Ratings
    "elo_ratings.pkl",
    
    # Player Props (se existir)
    "xgb_pts.joblib",
    "xgb_reb.joblib", 
    "xgb_ast.joblib",
}

def cleanup_models():
    """Remove modelos obsoletos."""
    removed = []
    kept = []
    backup_dir = MODELS_DIR / "backup_old"
    
    # Criar backup dir
    backup_dir.mkdir(exist_ok=True)
    
    for f in MODELS_DIR.glob("*.joblib"):
        if f.name in KEEP:
            kept.append(f.name)
        else:
            # Mover para backup em vez de deletar
            shutil.move(str(f), str(backup_dir / f.name))
            removed.append(f.name)
    
    for f in MODELS_DIR.glob("*.pkl"):
        if f.name in KEEP:
            kept.append(f.name)
        else:
            shutil.move(str(f), str(backup_dir / f.name))
            removed.append(f.name)
    
    print("=" * 60)
    print("🧹 LIMPEZA DE MODELOS")
    print("=" * 60)
    print(f"\n✅ Mantidos ({len(kept)}):")
    for k in sorted(kept):
        print(f"   - {k}")
    
    print(f"\n📦 Movidos para backup ({len(removed)}):")
    for r in sorted(removed)[:10]:
        print(f"   - {r}")
    if len(removed) > 10:
        print(f"   ... e mais {len(removed) - 10} arquivos")
    
    print(f"\n📂 Backup em: {backup_dir}")
    print("\n✅ Limpeza concluída!")
    
    return removed, kept

if __name__ == "__main__":
    cleanup_models()
