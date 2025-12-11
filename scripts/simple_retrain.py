#!/usr/bin/env python3
"""Script simplificado para retreinar o modelo ML."""
import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, '/home/denis/nba-predictor')

# Importar e executar treinamento
from ml_pipeline.train_ensemble import train_ensemble_model

if __name__ == "__main__":
    print("🎯 Iniciando retreinamento do modelo ML...")
    try:
        model, accuracy = train_ensemble_model()
        print(f"\n✅ Modelo retreinado com sucesso!")
        print(f"📊 Acurácia: {accuracy*100:.2f}%")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro no retreinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
