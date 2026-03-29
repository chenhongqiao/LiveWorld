#!/bin/bash
# LiveWorld Environment Setup
# Installs all dependencies and downloads model weights.
set -e

echo "============================================================"
echo "LiveWorld Environment Setup"
echo "============================================================"

# Step 1: Install Python dependencies
echo ""
echo ">>> Step 1/3: Install Python dependencies"
pip install -r setup/requirements.txt

# Step 2: Install LiveWorld + local packages (sam3, stream3r)
echo ""
echo ">>> Step 2/3: Install LiveWorld and local packages"
bash setup/install_packages.sh

# Step 3: Download model weights
echo ""
echo ">>> Step 3/3: Download model weights"
bash setup/download_ckpts.sh

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
