# Fixed upstream editable source preserves native tools/config path resolution.
# Trusted Agent controller only; generated code must execute in separate sandboxes.
ARG BASE_IMAGE=python@sha256:6cbe849bdc876ecd413a4a09a919a0d861d908589de9eba61e886c29af793a07
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY source/ /opt/SWE-agent/
WORKDIR /opt/SWE-agent
RUN python -m pip install --no-cache-dir --no-compile -e .
# LiteLLM imports its MCP bridge during native model calls, not during CLI help.
# Install its declared optional dependencies rather than relying on import smoke.
RUN python -m pip install --no-cache-dir --no-compile 'litellm[mcp]==1.108.0' && python -m pip check
ENV PYTHONDONTWRITEBYTECODE=1
