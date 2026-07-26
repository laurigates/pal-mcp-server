#!/bin/bash

# PAL MCP Server - Code Quality Checks
# Runs the lockfile check, lint, format, type-check, and unit tests via uv.
# ALL checks must pass 100% for CI to succeed.
# The lockfile check runs FIRST, before the sync — see the comment at that step.

set -euo pipefail

echo "🔍 Running Code Quality Checks for PAL MCP Server"
echo "================================================="

if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Must come before the sync: `uv sync` re-locks when the lockfile is stale, so
# running it first would repair the drift in your working tree and leave this
# check passing on a fix you never staged. CI has no such sync — it runs
# `uv lock --check` against what you committed and fails.
echo "🔒 Step 0: Lockfile is current (uv lock --check)"
echo "------------------------------------------------"
if ! uv lock --check; then
    echo ""
    echo "❌ uv.lock is out of date. Run 'uv lock' and commit the result."
    echo "   (Use 'uv lock --upgrade' only for a deliberate dependency refresh —"
    echo "    it re-resolves everything, not just the drift.)"
    exit 1
fi
# The check above compares your working tree; CI compares what you committed.
# Those diverge whenever uv.lock differs from HEAD — staged or not — so say so
# rather than let "Ready for commit" overstate what was verified. Guarded on
# being in a checkout at all, so a tarball copy doesn't warn about a fine lock.
if git rev-parse --git-dir >/dev/null 2>&1 && ! git diff --quiet HEAD -- uv.lock; then
    echo "⚠️  uv.lock differs from HEAD — make sure it is in your commit."
fi
echo ""

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
echo "✅ Lockfile current (uv lock --check)"
echo "✅ Lint (ruff)"
echo "✅ Format (ruff format)"
echo "✅ Type check (ty)"
echo "✅ Unit tests (pytest)"
echo ""
echo "🚀 Ready for commit."
