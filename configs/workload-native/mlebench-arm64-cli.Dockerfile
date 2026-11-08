# CLI-only ARM64 profile, not an Agent/scientific execution environment.
# Context: verified author/ plus this recipe and mlebench-arm64-client-py312.txt.
# Reuse the qualified existing Python3.12 Ascend base without exposing NPU devices.
FROM quay.io/ascend/vllm-ascend@sha256:105834a38766a6b1b89a7eeb313a37351d098a69e8cdee87ad0ca3a6e090ce13

USER root
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="" \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

WORKDIR /opt/mlebench-build

COPY mlebench-arm64-client-py312.txt /opt/mlebench-requirements.txt
RUN /usr/local/python3.12.13/bin/python3 -m venv /opt/mlebench-client \
    && /opt/mlebench-client/bin/python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple --report /opt/mlebench-pip-report.json \
    -r /opt/mlebench-requirements.txt \
    && /opt/mlebench-client/bin/python -m pip check

COPY author /opt/mlebench-author
RUN /opt/mlebench-client/bin/python -m pip install --no-cache-dir \
    --no-deps --no-build-isolation /opt/mlebench-author \
    && /opt/mlebench-client/bin/python -m pip check \
    && HOME=/tmp /opt/mlebench-client/bin/mlebench --help > /opt/mlebench-cli-help.txt

ENV PATH="/opt/mlebench-client/bin:${PATH}"
# The original CLI creates diskcache("cache") even for --help.
WORKDIR /tmp
ENTRYPOINT ["/opt/mlebench-client/bin/mlebench"]
CMD ["--help"]
