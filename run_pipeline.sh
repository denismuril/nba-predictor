#!/bin/bash
# Wrapper script to run NBA Predictor pipeline
# This ensures the virtual environment is verified

cd /home/denis/nba-predictor
source venv/bin/activate

# Execute the main pipeline command
# Adjust this command to whatever your main entry point is
python -m ml_pipeline.train_all_models
