#!/usr/bin/env python3
"""Query rápida para verificar dados V21 no banco"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

db = get_db_manager()
preds = db.get_latest_predictions('2025-12-08')

print(f"\n📊 Total previsões: {len(preds)}\n")

if not preds.empty:
    # Verificar se colunas V21 existem
    v21_cols = ['home_shooting_luck', 'away_shooting_luck', 'home_rapm_penalty', 'away_rapm_penalty']
    
    print("🔍 Colunas V21 no DataFrame:")
    for col in v21_cols:
        exists = col in preds.columns
        print(f"   {col}: {'✅ EXISTS' if exists else '❌ MISSING'}")
    
    print("\n📋 Primeira previsão:")
    first = preds.iloc[0]
    print(f"   {first['home_team']} vs {first['away_team']}")
    
    if 'home_shooting_luck' in preds.columns:
        print(f"   home_shooting_luck: {first['home_shooting_luck']}")
        print(f"   away_shooting_luck: {first['away_shooting_luck']}")
        print(f"   home_rapm_penalty: {first['home_rapm_penalty']}")
        print(f"   away_rapm_penalty: {first['away_rapm_penalty']}")
    else:
        print("   ⚠️ Colunas V21 não encontradas!")
