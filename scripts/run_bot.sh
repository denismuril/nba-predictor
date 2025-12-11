#!/bin/bash

# Configuration
PROJECT_DIR="/home/denis/nba-predictor"
BOT_SCRIPT="telegram_bot/nba_tigrinho_bot.py"
LOG_FILE="$PROJECT_DIR/logs/bot_service.log"
LOCK_FILE="$PROJECT_DIR/logs/bot.lock"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/logs"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🚀 Starting NBA Predictor Bot Service..."

# Navigate to project directory
cd "$PROJECT_DIR" || { log "❌ Failed to change directory to $PROJECT_DIR"; exit 1; }

# Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    log "❌ Virtual environment not found!"
    exit 1
fi

# Check if already running
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if ps -p "$PID" > /dev/null; then
        log "⚠️ Bot is already running with PID $PID. Exiting."
        exit 0
    else
        log "⚠️ Lock file exists but process $PID is not running. Removing lock."
        rm "$LOCK_FILE"
    fi
fi

# Main Loop for Auto-Restart
while true; do
    log "🤖 Starting Bot..."
    
    # Create lock file
    echo $$ > "$LOCK_FILE"
    
    # Run the bot
    python "$BOT_SCRIPT" >> "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    log "⚠️ Bot crashed with exit code $EXIT_CODE. Restarting in 5 seconds..."
    
    # Remove lock file on crash
    rm -f "$LOCK_FILE"
    
    sleep 5
done
