#!/usr/bin/env python3
"""Script para atualizar resultados dos jogos dos dias 24 e 25"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager

print("🔄 Atualizando resultados dos jogos...")

db = get_db_manager()
df = db.get_history()

# Verificar jogos pendentes dos dias 24 e 25
target_dates = ['2025-11-24', '2025-11-25']
for date in target_dates:
    games = df[df['date'].astype(str).str.startswith(date)]
    pending = games[games['winner'].isnull()]
    print(f"📅 {date}: {len(games)} jogos total, {len(pending)} pendentes")

print("\n🔄 Executando atualização...")
updated = db.update_pending_results()
print(f"✅ {updated} jogos atualizados com sucesso!")
