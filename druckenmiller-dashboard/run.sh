#!/bin/bash
# Quick start script for the Druckenmiller Trading Dashboard

echo "Starting Druckenmiller Trading Dashboard..."
echo "Make sure you've installed dependencies: pip install -r requirements.txt"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Run Streamlit
streamlit run app/dashboard.py
