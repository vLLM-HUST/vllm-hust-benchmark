# Build context: configs/workload-native (no datasets or model weights).
FROM python@sha256:6cbe849bdc876ecd413a4a09a919a0d861d908589de9eba61e886c29af793a07
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    NLTK_DATA=/opt/nltk_data
COPY webarena-agentlab-0.4.2-official-constraints.txt /opt/official-constraints.txt
RUN python -m pip install -c /opt/official-constraints.txt agentlab==0.4.2 browsergym-webarena==0.14.2 \
    && python -m pip check
RUN python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip
RUN python -m nltk.downloader -d /opt/nltk_data punkt_tab \
    && python -c "import hashlib; from pathlib import Path; assert hashlib.sha256(Path('/opt/nltk_data/tokenizers/punkt_tab.zip').read_bytes()).hexdigest() == 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106'" # pragma: allowlist secret - public NLTK resource content identity
WORKDIR /work
