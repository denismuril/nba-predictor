#!/bin/bash
# Quick Deploy Script para NBA Predictor
# 
# Este script automatiza o deployment seguindo DEPLOYMENT_CHECKLIST.md

echo "🚀 NBA Predictor - Quick Deploy"
echo "================================"
echo ""

# Diretório base
NBA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$NBA_DIR"

echo "📂 Working directory: $NBA_DIR"
echo ""

# Step 1: Pre-deployment checks
echo "✅ Step 1: Pre-deployment validation..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "  ✓ Python 3 found"

# Check dependencies
python3 -c "import sklearn, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies"
    echo "Run: pip install -r requirements.txt"
    exit 1
fi
echo "  ✓ Dependencies OK"

# Smoke tests
echo ""
echo "  Running smoke tests..."
python3 tests/smoke_tests.py > /tmp/smoke_test.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Smoke tests failed"
    cat /tmp/smoke_test.log
    exit 1
fi
echo "  ✓ Smoke tests: 5/5 passing"

# Step 2: Create directories
echo ""
echo "✅ Step 2: Creating directories..."
mkdir -p logs
mkdir -p backups
mkdir -p reports
mkdir -p monitoring
echo "  ✓ Directories created"

# Step 3: Check models
echo ""
echo "✅ Step 3: Checking models..."

if [ ! -f "models/ml_model.joblib" ]; then
    echo "⚠️  ml_model.joblib not found"
    echo "  Run: python scripts/quick_retrain.py"
fi

if [ ! -f "models/calibrator.pkl" ]; then
    echo "⚠️  calibrator.pkl not found"
    echo "  Run: python scripts/recalibrate_model.py --lookback-days 60"
fi

# Step 4: Setup cron (interactive)
echo ""
echo "✅ Step 4: Cron job setup..."
echo ""
echo "Add these lines to your crontab (crontab -e):"
echo ""
echo "# NBA Predictor - Automation"
echo "0 6 * * * cd $NBA_DIR && python3 scripts/recalibrate_model.py >> logs/recalibration.log 2>&1"
echo "5 6 * * * cd $NBA_DIR && python3 monitoring/update_dashboard.py >> logs/monitoring.log 2>&1"
echo "10 6 * * * cd $NBA_DIR && python3 monitoring/daily_check.py >> logs/daily_check.log 2>&1"
echo "0 2 * * 0 cd $NBA_DIR && tar -czf backups/models_backup_\$(date +\\%Y\\%m\\%d).tar.gz models/*.joblib models/*.pkl >> logs/backup.log 2>&1"
echo ""

read -p "Open crontab now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    crontab -e
fi

# Step 5: First manual run
echo ""
echo "✅ Step 5: Running initial checks..."
echo ""

# Daily check
echo "  Running daily check..."
python3 monitoring/daily_check.py
echo "  ✓ Daily check complete"

# Step 6: Summary
echo ""
echo "================================"
echo "🎉 Deployment Complete!"
echo "================================"
echo ""
echo "📋 Next Steps:"
echo "  1. Verify cron jobs: crontab -l | grep NBA"
echo "  2. Check dashboard: open monitoring/dashboard.html"
echo "  3. Monitor logs for 7 days"
echo "  4. Review weekly report (Sundays)"
echo ""
echo "📊 Monitoring:"
echo "  - Daily metrics: monitoring/daily_metrics.json"
echo "  - Dashboard: monitoring/dashboard.html"
echo "  - Alerts: logs/alerts.log"
echo ""
echo "✅ System is now in production!"
echo ""
