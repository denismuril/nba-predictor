"""
Script para executar verificações de qualidade de código (Linting & Formatting).

Ferramentas:
- Black: Formatação de código
- Isort: Ordenação de imports
- Flake8: Análise estática de erros e estilo

Usage:
    python scripts/lint.py [--check] [--fix]
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Executa um comando de shell e loga o resultado."""
    logger.info(f"\n🚀 Executando {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} passou!")
            return True
        else:
            logger.error(f"❌ {description} falhou:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro executando {description}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Code Quality Tools')
    parser.add_argument('--check', action='store_true', help='Apenas verificar (sem modificar)')
    parser.add_argument('--fix', action='store_true', help='Corrigir automaticamente (Black/Isort)')
    args = parser.parse_args()
    
    # Se nenhum argumento, assume --check
    if not args.fix:
        args.check = True
        
    project_root = "."
    
    success = True
    
    # 1. Isort (Imports)
    if args.fix:
        if not run_command(f"{sys.executable} -m isort {project_root}", "Isort (Fix)"):
            success = False
    else:
        if not run_command(f"{sys.executable} -m isort --check-only --diff {project_root}", "Isort (Check)"):
            success = False
            
    # 2. Black (Formatting)
    if args.fix:
        if not run_command(f"{sys.executable} -m black {project_root}", "Black (Fix)"):
            success = False
    else:
        if not run_command(f"{sys.executable} -m black --check --diff {project_root}", "Black (Check)"):
            success = False
            
    # 3. Flake8 (Linting) - sempre check
    if not run_command(f"{sys.executable} -m flake8 {project_root}", "Flake8 (Linting)"):
        success = False
        
    if success:
        logger.info("\n✨ TUDO LIMPO! Código aprovado.")
        return 0
    else:
        logger.error("\n⚠️  Algumas verificações falharam.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
