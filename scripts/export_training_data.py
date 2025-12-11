#!/usr/bin/env python3
"""Script para exportar dados históricos do banco para CSV."""
import sys
import pandas as pd
sys.path.insert(0, '/home/denis/nba-predictor')

from data.repositories.db_manager import get_db_manager

def export_historical_data():
    """Exporta dados históricos para CSV."""
    print("📊 Exportando dados históricos do banco...")
    
    try:
        db = get_db_manager()
        df = db.get_comprehensive_history()
        
        if df is None or df.empty:
            print("❌ Nenhum dado histórico encontrado")
            return False
        
        output_path = '/tmp/nba_training_data.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✅ {len(df)} registros exportados para {output_path}")
        print(f"📋 Colunas: {list(df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao exportar dados: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = export_historical_data()
    sys.exit(0 if success else 1)
