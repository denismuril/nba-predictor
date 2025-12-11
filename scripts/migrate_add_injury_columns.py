#!/usr/bin/env python3
"""
Script to add injury columns to predictions table
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = DatabaseManager()
conn = db.get_connection()

try:
    cursor = conn.cursor()
    
    # Attempt to add columns (will fail if they already exist, which is fine)
    try:
        logger.info("Adding home_injuries_list column...")
        cursor.execute(db._prepare_query("ALTER TABLE predictions ADD COLUMN home_injuries_list TEXT"))
        logger.info("✅ home_injuries_list added!")
    except Exception as e:
        logger.info(f"Column home_injuries_list might already exist: {e}")
    
    try:
        logger.info("Adding away_injuries_list column...")
        cursor.execute(db._prepare_query("ALTER TABLE predictions ADD COLUMN away_injuries_list TEXT"))
        logger.info("✅ away_injuries_list added!")
    except Exception as e:
        logger.info(f"Column away_injuries_list might already exist: {e}")
    
    conn.commit()
    logger.info("✅ Migration complete!")
    
except Exception as e:
    conn.rollback()
    logger.error(f"❌ Migration failed: {e}")
finally:
    db.return_connection(conn)
