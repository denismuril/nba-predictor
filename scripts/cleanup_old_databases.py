"""
Script de Limpeza de Arquivos de Banco de Dados

Remove arquivos .db temporários e consolida databases de apostas.

ATENÇÃO: Este script deleta arquivos! Use com cuidado.

Usage:
    python scripts/cleanup_old_databases.py [--dry-run] [--force]
"""
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseCleaner:
    """Gerenciador de limpeza de arquivos de banco de dados."""
    
    def __init__(self, data_dir='data', dry_run=False):
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {data_dir}")
    
    def find_databases(self):
        """Encontra todos arquivos .db no diretório."""
        db_files = list(self.data_dir.glob('*.db'))
        logger.info(f"📊 Encontrados {len(db_files)} arquivos .db")
        return db_files
    
    def categorize_files(self, db_files):
        """Categoriza arquivos por tipo."""
        categories = {
            'temp': [],        # Arquivos temporários
            'backup': [],      # Arquivos de backup
            'bets': [],        # Databases de apostas
            'active': [],      # Databases ativos
            'unknown': []      # Outros
        }
        
        for db_file in db_files:
            name = db_file.name.lower()
            
            if 'temp' in name or '_temp_' in name:
                categories['temp'].append(db_file)
            elif 'backup' in name:
                categories['backup'].append(db_file)
            elif 'bet' in name:  # bets.db, betting_tracker.db, backtest_bets.db
                categories['bets'].append(db_file)
            elif name in ['nba_history.db', 'nba_predictions.db', 'nba_predictor.db']:
                categories['active'].append(db_file)
            else:
                categories['unknown'].append(db_file)
        
        return categories
    
    def show_analysis(self, categories):
        """Mostra análise dos arquivos encontrados."""
        logger.info("\n" + "="*80)
        logger.info("📋 ANÁLISE DE ARQUIVOS")
        logger.info("="*80)
        
        for cat_name, files in categories.items():
            if files:
                logger.info(f"\n{cat_name.upper()}: {len(files)} arquivo(s)")
                for f in files:
                    size_mb = f.stat().st_size / 1024 / 1024
                    logger.info(f"  - {f.name} ({size_mb:.2f} MB)")
    
    def cleanup_temp_files(self, temp_files):
        """Remove arquivos temporários."""
        if not temp_files:
            logger.info("\n✅ Nenhum arquivo temporário para limpar")
            return 0
        
        logger.info(f"\n🗑️  Limpando {len(temp_files)} arquivo(s) temporário(s)...")
        
        deleted = 0
        for temp_file in temp_files:
            if self.dry_run:
                logger.info(f"  [DRY-RUN] Deletaria: {temp_file.name}")
                deleted += 1
            else:
                try:
                    temp_file.unlink()
                    logger.info(f"  ✅ Deletado: {temp_file.name}")
                    deleted += 1
                except Exception as e:
                    logger.error(f"  ❌ Erro ao deletar {temp_file.name}: {e}")
        
        return deleted
    
    def consolidate_bets_databases(self, bets_files):
        """Consolida databases de apostas."""
        if len(bets_files) <= 1:
            logger.info("\n✅ Apenas 1 database de apostas - não precisa consolidar")
            return 0
        
        logger.info(f"\n🔄 Encontrados {len(bets_files)} databases de apostas:")
        for f in bets_files:
            logger.info(f"  - {f.name}")
        
        # Identificar o database principal (bets.db geralmente é o principal)
        main_db = None
        for f in bets_files:
            if f.name == 'bets.db':
                main_db = f
                break
        
        if not main_db:
            main_db = bets_files[0]  # Usar o primeiro como principal
        
        logger.info(f"\n📌 Database principal: {main_db.name}")
        logger.info("⚠️  ATENÇÃO: Consolidação de databases requer migração manual dos dados")
        logger.info("   Recomendação:")
        logger.info(f"   1. Manter apenas: {main_db.name}")
        logger.info("   2. Outros databases:")
        
        renamed = 0
        for f in bets_files:
            if f != main_db:
                # Renomear para .bak ao invés de deletar
                backup_name = f.with_suffix('.db.bak')
                
                if self.dry_run:
                    logger.info(f"      [DRY-RUN] Renomearia: {f.name} -> {backup_name.name}")
                    renamed += 1
                else:
                    try:
                        f.rename(backup_name)
                        logger.info(f"      ✅ Renomeado: {f.name} -> {backup_name.name}")
                        renamed += 1
                    except Exception as e:
                        logger.error(f"      ❌ Erro: {e}")
        
        return renamed
    
    def cleanup_old_backups(self, backup_files, keep_recent=3):
        """Remove backups antigos, mantendo os N mais recentes."""
        if len(backup_files) <= keep_recent:
            logger.info(f"\n✅ {len(backup_files)} backup(s) - dentro do limite de {keep_recent}")
            return 0
        
        # Ordenar por data de modificação (mais recentes primeiro)
        sorted_backups = sorted(backup_files, key=lambda f: f.stat().st_mtime, reverse=True)
        
        to_keep = sorted_backups[:keep_recent]
        to_delete = sorted_backups[keep_recent:]
        
        logger.info(f"\n🗑️  Limpando backups antigos (mantendo {keep_recent} mais recentes)...")
        logger.info("Mantendo:")
        for f in to_keep:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            logger.info(f"  ✅ {f.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
        
        deleted = 0
        logger.info("\nDeletando:")
        for f in to_delete:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            
            if self.dry_run:
                logger.info(f"  [DRY-RUN] Deletaria: {f.name} ({mtime.strftime('%Y-%m-%d')})")
                deleted += 1
            else:
                try:
                    f.unlink()
                    logger.info(f"  ✅ Deletado: {f.name} ({mtime.strftime('%Y-%m-%d')})")
                    deleted += 1
                except Exception as e:
                    logger.error(f"  ❌ Erro: {e}")
        
        return deleted


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Limpeza de arquivos de banco de dados')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simular limpeza sem deletar arquivos')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Diretório de dados (default: data)')
    parser.add_argument('--keep-backups', type=int, default=3,
                       help='Número de backups a manter (default: 3)')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🔍 MODO DRY-RUN - Nenhum arquivo será modificado\n")
    
    try:
        cleaner = DatabaseCleaner(data_dir=args.data_dir, dry_run=args.dry_run)
        
        # 1. Encontrar databases
        db_files = cleaner.find_databases()
        
        # 2. Categorizar
        categories = cleaner.categorize_files(db_files)
        
        # 3. Mostrar análise
        cleaner.show_analysis(categories)
        
        # 4. Limpeza
        logger.info("\n" + "="*80)
        logger.info("🧹 EXECUTANDO LIMPEZA")
        logger.info("="*80)
        
        temp_deleted = cleaner.cleanup_temp_files(categories['temp'])
        backups_deleted = cleaner.cleanup_old_backups(categories['backup'], args.keep_backups)
        bets_consolidated = cleaner.consolidate_bets_databases(categories['bets'])
        
        # 5. Resumo
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMO")
        logger.info("="*80)
        logger.info(f"Arquivos temporários deletados: {temp_deleted}")
        logger.info(f"Backups antigos deletados: {backups_deleted}")
        logger.info(f"Databases de apostas consolidados: {bets_consolidated}")
        
        if args.dry_run:
            logger.info("\n💡 Execute sem --dry-run para aplicar as mudanças")
        else:
            logger.info("\n✅ Limpeza concluída!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
