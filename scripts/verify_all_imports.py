#!/usr/bin/env python3
"""Final verification of all system imports after scraper reorganization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    print("Testing key imports...")
    results = []
    
    # 1. Props scrapers from new location
    try:
        from data.scrapers.props import ActionNetworkScraper, LinemateScraper, CoversScraper
        results.append("✅ Props scrapers from props/ directory")
    except ImportError as e:
        results.append(f"❌ Props scrapers: {e}")
    
    # 2. OddsDataManager 
    try:
        from data.odds_manager import OddsDataManager
        results.append("✅ OddsDataManager")
    except ImportError as e:
        results.append(f"❌ OddsDataManager: {e}")
    
    # 3. Player name normalizer with thefuzz
    try:
        from data.scrapers.player_name_normalizer import get_fuzzy_backend
        results.append(f"✅ Player name normalizer (backend: {get_fuzzy_backend()})")
    except ImportError as e:
        results.append(f"❌ Player name normalizer: {e}")
    
    # 4. Orchestrator
    try:
        from orchestrator import EnterpriseOrchestrator
        results.append("✅ EnterpriseOrchestrator")
    except ImportError as e:
        results.append(f"❌ EnterpriseOrchestrator: {e}")
    
    # 5. Training scripts
    try:
        from scripts.train_all_models import main as train_main
        results.append("✅ train_all_models")
    except ImportError as e:
        results.append(f"❌ train_all_models: {e}")
    
    # 6. Generate player props
    try:
        from scripts.generate_player_props_quick import generate_player_props
        results.append("✅ generate_player_props_quick")
    except ImportError as e:
        results.append(f"❌ generate_player_props_quick: {e}")
    
    # Print results
    print()
    for r in results:
        print(f"  {r}")
    
    # Summary
    errors = sum(1 for r in results if r.startswith("❌"))
    print()
    if errors == 0:
        print("🎉 All imports successful!")
        return 0
    else:
        print(f"⚠️ {errors} errors found")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
