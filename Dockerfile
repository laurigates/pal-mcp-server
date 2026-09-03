# ===========================================
# STAGE 1: Build dependencies with uv
# ===========================================
# Astral stopped publishing bookworm-slim variants after 0.9, which left this
# pin frozen on a discontinued image and gave Renovate nothing to bump. The
# trixie-slim line is the maintained one, so the pin now tracks it.
#
# Keep this on the same Debian release as the runtime stage below, and keep the
# version inside `[tool.uv] required-version` in pyproject.toml — the container
# is a third place that bound applies to, via `uv sync --frozen`.
FROM ghcr.io/astral-sh/uv:0.12.9-python3.12-trixie-slim AS builder

# Configure uv for reproducible, container-friendly installs:
# - link mode "copy" works on every filesystem (cache mounts are scoped)
# - bytecode compilation speeds up cold starts
# - python downloads disabled (we already have the interpreter from the base image)
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Install only the locked runtime dependencies first to maximise layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then install the project itself.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ===========================================
# STAGE 2: Runtime image
# ===========================================
# Named explicitly rather than via the floating `3.12-slim`, which resolves to
# the same digest today: the venv copied from the builder should land on the
# Debian release it was resolved on, and that only holds if both are stated.
FROM python:3.12-slim-trixie AS runtime

LABEL maintainer="PAL MCP Server Team"
LABEL org.opencontainers.image.title="pal-mcp-server"
LABEL org.opencontainers.image.description="AI-powered Model Context Protocol server with multi-provider support"
LABEL org.opencontainers.image.source="https://github.com/laurigates/pal-mcp-server"
LABEL org.opencontainers.image.documentation="https://github.com/laurigates/pal-mcp-server/blob/main/README.md"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Create non-root user for security
RUN groupadd -r paluser && useradd -r -g paluser paluser

# Install minimal runtime dependencies.
#
# `upgrade` is not cosmetic: the release job scans this image at CRITICAL,HIGH
# with unfixed findings ignored, and the base always lags trixie-security by
# whatever has been published since the base was last cut. How far it lags
# varies — measured 2026-08-28, a build off the then-current base still carried
# 3 HIGH in openssl/libssl3t64 (fixed in 3.5.7-1~deb13u2) that only this line
# clears, while the published 10.4.4 image, built off an older base, carried
# those plus 36 more across the util-linux family. No dependency change of ours
# reaches any of them.
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the resolved virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY --chown=paluser:paluser . .

# Create logs and tmp directories with proper permissions
RUN mkdir -p logs tmp && chown -R paluser:paluser logs tmp

# Copy health check script
COPY --chown=paluser:paluser docker/scripts/healthcheck.py /usr/local/bin/healthcheck.py
RUN chmod +x /usr/local/bin/healthcheck.py

USER paluser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /usr/local/bin/healthcheck.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["python", "server.py"]
