#!/usr/bin/env python3
"""Check if predictions have injury columns"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import DatabaseManager

db = DatabaseManager()
preds = db.get_latest_predictions()

print("=" * 80)
print("ANÁLISE DO BANCO DE DADOS")
print("=" * 80)

print(f"\nNúmero de previsões: {len(preds)}")
print(f"\nColunas disponíveis ({len(preds.columns)}):")
for i, col in enumerate(sorted(preds.columns), 1):
    print(f"  {i:3d}. {col}")

print("\n" + "=" * 80)
print("VERIFICAÇÃO DE COLUNAS DE LESÕES")
print("=" * 80)

injury_cols = [c for c in preds.columns if 'injur' in c.lower()]
print(f"\nColunas relacionadas a 'injury': {injury_cols}")

rapm_cols = [c for c in preds.columns if 'rapm' in c.lower()]
print(f"Colunas relacionadas a 'rapm': {rapm_cols}")

if len(preds) > 0:
    print("\n" + "=" * 80)
    print("EXEMPLO DE DADOS (primeira previsão)")
    print("=" * 80)
    
    row = preds.iloc[0]
    print(f"\nData: {row.get('date', 'N/A')}")
    print(f"Jogo: {row.get('away_team', '?')} @ {row.get('home_team', '?')}")
    
    print(f"\nRAPM Penalty:")
    print(f"  Home: {row.get('home_rapm_penalty', 'COLUNA NÃO EXISTE')}")
    print(f"  Away: {row.get('away_rapm_penalty', 'COLUNA NÃO EXISTE')}")
    
    print(f"\nInjuries List:")
    print(f"  Home: {row.get('home_injuries_list', 'COLUNA NÃO EXISTE')}")
    print(f"  Away: {row.get('away_injuries_list', 'COLUNA NÃO EXISTE')}")

print("\n✅ Análise concluída!")
