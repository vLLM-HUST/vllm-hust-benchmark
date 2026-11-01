# API transport and original grader only; inference stays on an external NPU.
# Build context must contain only this file and the public lock, never datasets.
FROM public.ecr.aws/docker/library/python@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 HF_HOME=/tmp/huggingface PYTHONDONTWRITEBYTECODE=1
COPY livecodebench-api-cpu.lock.txt /opt/livecodebench-api-cpu.lock.txt
RUN test "$(uname -m)" = aarch64 && \
    python -m pip install -r /opt/livecodebench-api-cpu.lock.txt && \
    python -m pip check && \
    python -c "import datasets, openai, anthropic, pebble, numpy; assert datasets.__version__ == '3.2.0'; assert openai.__version__ == '1.59.6'"
USER 65534:65534
WORKDIR /tmp
