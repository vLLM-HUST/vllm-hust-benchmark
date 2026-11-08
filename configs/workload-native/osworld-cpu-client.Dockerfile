# ARM64 CPython3.11 software client. No model weights, accelerator devices or VM image.
FROM python@sha256:6cbe849bdc876ecd413a4a09a919a0d861d908589de9eba61e886c29af793a07
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc6-dev qemu-system-x86 qemu-utils seabios \
    && rm -r -- /var/lib/apt/lists
COPY osworld-client-requirements.lock /opt/osworld-client-requirements.lock
RUN python -m pip install --no-cache-dir -r /opt/osworld-client-requirements.lock \
    && python -m pip check \
    && python -m pip freeze > /opt/osworld-client-installed.txt
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /tmp
