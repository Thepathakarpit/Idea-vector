#!/bin/bash

# Update package list and install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv build-essential

# Install Python dependencies
pip install google-generativeai numpy scikit-learn matplotlib

echo "All dependencies installed successfully!"