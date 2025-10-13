#!/bin/bash

# Nyanta Startup Script
echo "Starting Nyanta..."
echo ""

# Check if venv exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Error: venv not found. Run: python3.11 -m venv venv"
    exit 1
fi

# Verify correct Python
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run application
python -m streamlit run app.py