#!/bin/bash
set -e

echo "Installing dependencies with pip..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "✅ Dependencies installed successfully!"

