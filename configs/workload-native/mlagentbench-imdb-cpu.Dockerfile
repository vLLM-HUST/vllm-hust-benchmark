# Task-only scientific CPU execution. Agent LLM requests use the NPU queue.
# Build context: wheels/ and requirements-verified.txt from the companion ARM lock.
ARG BASE_IMAGE=python:3.11-slim-bookworm@sha256:6cbe849bdc876ecd413a4a09a919a0d861d908589de9eba61e886c29af793a07
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-verified.txt /opt/imdb-client/requirements-verified.txt
COPY wheels/ /opt/imdb-client/wheels/
RUN python -m pip install --no-index --no-deps --require-hashes --find-links /opt/imdb-client/wheels -r /opt/imdb-client/requirements-verified.txt && python -m pip check
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
WORKDIR /run
