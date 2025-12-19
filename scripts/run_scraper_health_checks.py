#!/usr/bin/env python3
"""
Scraper Health Check Script - NBA Predictor

Tests connectivity and functionality for all player props scrapers.
Run: python scripts/run_scraper_health_checks.py
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class HealthCheckRunner:
    """Runs health checks on all scrapers."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.today = datetime.now().strftime("%Y-%m-%d")
    
    async def check_scraper(
        self, 
        name: str, 
        scraper_class: type, 
        method_name: str = "get_props"
    ) -> Dict[str, Any]:
        """
        Tests a single scraper.
        
        Returns:
            Dict with status, timing, and error info
        """
        result = {
            "status": "unknown",
            "props_count": 0,
            "time_seconds": 0.0,
            "error": None,
            "prop_types": [],
        }
        
        print(f"\n🔄 Testing {name}...", end=" ", flush=True)
        
        start = time.time()
        
        try:
            scraper = scraper_class(headless=True)
            
            # Try to fetch props
            method = getattr(scraper, method_name, None)
            if method is None:
                method = getattr(scraper, "fetch_props", None)
            
            if method is None:
                result["status"] = "no_method"
                result["error"] = f"No {method_name} or fetch_props method found"
                print("⚠️ NO METHOD")
                return result
            
            # Execute with timeout
            props = await asyncio.wait_for(method(self.today), timeout=60)
            
            result["time_seconds"] = round(time.time() - start, 2)
            result["props_count"] = len(props) if props else 0
            
            if props and len(props) > 0:
                result["status"] = "success"
                # Get unique prop types
                prop_types = list(set(p.prop_type for p in props if hasattr(p, 'prop_type')))
                result["prop_types"] = prop_types[:5]  # First 5
                print(f"✅ {result['props_count']} props in {result['time_seconds']}s")
            else:
                result["status"] = "empty"
                print(f"⚠️ EMPTY in {result['time_seconds']}s")
                
        except asyncio.TimeoutError:
            result["status"] = "timeout"
            result["error"] = "Timeout after 60s"
            result["time_seconds"] = 60.0
            print("⏰ TIMEOUT")
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = f"{type(e).__name__}: {str(e)[:100]}"
            result["time_seconds"] = round(time.time() - start, 2)
            print(f"❌ ERROR: {type(e).__name__}")
            
        return result
    
    async def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Runs health checks on all available scrapers."""
        
        scrapers: List[Tuple[str, str, str]] = [
            # (display_name, module_path, class_name)
            ("ActionNetwork", "data.scrapers.props.action_network_scraper", "ActionNetworkScraper"),
            ("Linemate", "data.scrapers.props.linemate_scraper", "LinemateScraper"),
            ("BettingPros", "data.scrapers.props.bettingpros_scraper", "BettingProsScraper"),
            ("Covers", "data.scrapers.props.covers_scraper", "CoversScraper"),
            ("PropsMadness", "data.scrapers.props.propsmadness_scraper", "PropsMadnessScraper"),
            ("PropsComScraper", "data.scrapers.props.propscom_scraper", "PropsComScraper"),
            ("LineStar", "data.scrapers.props.linestar_scraper", "LineStarScraper"),
        ]
        
        print("=" * 60)
        print("  SCRAPER HEALTH CHECK - NBA Predictor")
        print("=" * 60)
        print(f"  Date: {self.today}")
        print("=" * 60)
        
        for name, module_path, class_name in scrapers:
            try:
                # Dynamic import
                import importlib
                module = importlib.import_module(module_path)
                scraper_class = getattr(module, class_name)
                
                self.results[name] = await self.check_scraper(name, scraper_class)
                
            except ImportError as e:
                self.results[name] = {
                    "status": "import_error",
                    "error": f"Could not import: {e}",
                    "props_count": 0,
                    "time_seconds": 0,
                }
                print(f"\n🔄 Testing {name}... ❌ IMPORT ERROR")
                
            except Exception as e:
                self.results[name] = {
                    "status": "unexpected_error",
                    "error": f"{type(e).__name__}: {str(e)[:100]}",
                    "props_count": 0,
                    "time_seconds": 0,
                }
                print(f"\n🔄 Testing {name}... ❌ UNEXPECTED ERROR")
        
        return self.results
    
    def print_summary(self):
        """Prints a formatted summary of results."""
        
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        
        success_count = 0
        total_props = 0
        
        for name, result in self.results.items():
            status = result["status"]
            icon = {
                "success": "✅",
                "empty": "⚠️",
                "error": "❌",
                "timeout": "⏰",
                "import_error": "🚫",
                "no_method": "❓",
            }.get(status, "❓")
            
            count = result.get("props_count", 0)
            time_s = result.get("time_seconds", 0)
            
            if status == "success":
                success_count += 1
                total_props += count
            
            print(f"  {icon} {name:15} | Status: {status:12} | Props: {count:4} | Time: {time_s:.1f}s")
            
            if result.get("error"):
                print(f"      └─ Error: {result['error'][:50]}...")
            
            if result.get("prop_types"):
                print(f"      └─ Types: {', '.join(result['prop_types'])}")
        
        print("\n" + "-" * 60)
        print(f"  Working Scrapers: {success_count}/{len(self.results)}")
        print(f"  Total Props Found: {total_props}")
        print("=" * 60)
        
        # Write results to log file
        log_path = project_root / "logs" / "scraper_health_check.log"
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, "a") as f:
            f.write(f"\n--- Health Check {datetime.now().isoformat()} ---\n")
            for name, result in self.results.items():
                f.write(f"{name}: {result['status']} ({result.get('props_count', 0)} props)\n")
                if result.get("error"):
                    f.write(f"  Error: {result['error']}\n")
        
        print(f"\n📝 Results logged to: {log_path}")


async def main():
    """Main entry point."""
    runner = HealthCheckRunner()
    await runner.run_all_checks()
    runner.print_summary()
    
    # Return exit code based on results
    working = sum(1 for r in runner.results.values() if r["status"] == "success")
    return 0 if working > 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Cancelled by user.")
        sys.exit(1)
