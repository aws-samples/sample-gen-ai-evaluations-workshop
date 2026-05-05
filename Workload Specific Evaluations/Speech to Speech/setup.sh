#!/bin/bash
# =============================================================================
# SageMaker Studio Setup Script for S2S Evaluation Pipeline
#
# Before running this script:
#   1. cd ~/sample-gen-ai-evaluations-workshop/04-workload-specific-evaluations/04-05-Speech-to-Speech
#   2. python3 -m venv venv && source venv/bin/activate
#   3. cp .env.example .env && nano .env   # Add your AWS credentials
#   4. bash setup.sh
# =============================================================================
set -e

MODULE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$MODULE_DIR"

echo "============================================"
echo "  S2S Evaluation Pipeline — SageMaker Setup"
echo "============================================"
echo ""

# --- Step 1: Install Python dependencies ---
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt
pip install -q -r sample_s2s_app/python-server/requirements.txt
echo "✅ Python dependencies installed"
echo ""

# --- Step 2: Configure backend .env (AWS creds + OTel config) ---
echo "📝 Setting up backend environment..."
cp .env sample_s2s_app/python-server/.env
cat sample_s2s_app/python-server/env.example >> sample_s2s_app/python-server/.env
echo "   Created backend .env with AWS credentials and OTel config"

# --- Step 3: Create .env.test for Playwright ---
cp -n env.example .env.test 2>/dev/null || true
mkdir -p test
cp .env.test test/.env.test
echo "   Created .env.test and test/.env.test"

# --- Step 4: Generate SageMaker proxy URLs ---
echo ""
echo "🔗 Detecting SageMaker Studio URLs..."
python sagemaker_helper.py
echo ""

# --- Step 5: Build the React frontend ---
echo "⚛️  Building React frontend..."
cd sample_s2s_app/react-client
npm install --silent 2>&1 | tail -1
npm run build 2>&1 | tail -3
cd "$MODULE_DIR"
echo "✅ React frontend built"
echo ""

# --- Step 6: Install Playwright + Chromium ---
echo "🎭 Installing Playwright and Chromium..."
npm install --silent 2>&1 | tail -1
npx playwright install chromium 2>&1 | tail -3
echo "📦 Installing Chromium OS dependencies..."
npx playwright install-deps chromium 2>&1 | tail -5
echo "✅ Playwright ready"
echo ""

# --- Done ---
echo "============================================"
echo "  ✅ Setup complete!"
echo "============================================"
echo ""
echo "  Open the notebook and run all cells:"
echo "     s2s_entire_eval_pipeline.ipynb"
echo ""
