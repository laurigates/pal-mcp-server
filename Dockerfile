# ===========================================
# STAGE 1: Build dependencies with uv
# ===========================================
# Pinned to the uv this unversioned tag already resolves to. Astral stopped
# publishing bookworm-slim variants after 0.9 — `0.10-`/`0.11-python3.12-bookworm-slim`
# do not exist — so `:python3.12-bookworm-slim` is frozen at 0.9.30 (built
# 2026-02-04) rather than tracking latest. Stating that explicitly keeps
# `[tool.uv] required-version` in pyproject.toml honest: the container's uv is
# a third place that bound applies to, via `uv sync --frozen` below.
#
# This variant is no longer maintained. Migrating to the trixie-slim line
# (`0.11-python3.12-trixie-slim`) is the real fix and would align the container
# with CI's uv, but container.yml only builds on the release PR and tag push, so
# it cannot be verified here. Tracked separately.
FROM ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim AS builder

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
FROM python:3.12-slim AS runtime

LABEL maintainer="PAL MCP Server Team"
LABEL org.opencontainers.image.title="pal-mcp-server"
LABEL org.opencontainers.image.description="AI-powered Model Context Protocol server with multi-provider support"
LABEL org.opencontainers.image.source="https://github.com/laurigates/pal-mcp-server"
LABEL org.opencontainers.image.documentation="https://github.com/laurigates/pal-mcp-server/blob/main/README.md"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Create non-root user for security
RUN groupadd -r paluser && useradd -r -g paluser paluser

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
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
