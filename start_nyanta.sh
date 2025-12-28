#!/bin/bash

# Nyanta Startup Script
echo "Starting Nyanta..."
echo ""

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "⚠️ venv not found. Attempting to create it..."
    python3.11 -m venv venv
    if [ $? -eq 0 ]; then
        echo "✅ venv created."
        source venv/bin/activate
        pip install -r requirements.txt
    else
        echo "❌ Error: Could not create venv. check your python3.11 installation."
        exit 1
    fi
fi

# Verify correct Python
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run application
python -m streamlit run app.py