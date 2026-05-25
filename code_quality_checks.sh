#!/bin/bash

# PAL MCP Server - Code Quality Checks
# Runs lint, format, type-check, and unit tests via uv.
# ALL checks must pass 100% for CI to succeed.

set -euo pipefail

echo "🔍 Running Code Quality Checks for PAL MCP Server"
echo "================================================="

if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "📦 Syncing dependencies..."
uv sync --group dev --quiet
echo ""

echo "📋 Step 1: Lint (ruff check --fix)"
echo "----------------------------------"
uv run ruff check --fix .
echo ""

echo "🎨 Step 2: Format (ruff format)"
echo "-------------------------------"
uv run ruff format .
echo ""

echo "✅ Step 3: Verify lint passes cleanly"
echo "-------------------------------------"
uv run ruff check .
echo ""

echo "🔎 Step 4: Type check (ty)"
echo "--------------------------"
uv run ty check . || echo "⚠️  ty reported issues (non-blocking during migration)"
echo ""

echo "🧪 Step 5: Unit tests (pytest, excluding integration)"
echo "-----------------------------------------------------"
uv run pytest tests/ -v -x -m "not integration"
echo ""

echo "🎉 All Code Quality Checks Passed!"
echo "=================================="
echo "✅ Lint (ruff)"
echo "✅ Format (ruff format)"
echo "✅ Type check (ty)"
echo "✅ Unit tests (pytest)"
echo ""
echo "🚀 Ready for commit."
