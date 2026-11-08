# ARM64 counterpart of the fixed author's dockerfile (Git blob c4ab072cd3f12c36a475ced33a9c863db0423bfb).
# Official upstream bases are pinned to metadata-verified ARM64 manifests.
FROM ghcr.io/astral-sh/uv@sha256:43b6b8ab14b48e7850828739e984661755f25cb48b5f3a812d846df8846ff5cf AS uv
FROM public.ecr.aws/docker/library/python@sha256:8f51ccb8569b75cf967ffe3e1638b9953042bf36142d7032cbb6f0696c9cd2c6
ENV DEBIAN_FRONTEND=noninteractive VIRTUAL_ENV=/opt/venv \
    APPWORLD_ROOT=/work APPWORLD_CACHE=/opt/appworld-cache \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=uv /uv /uv
# psutil 5.9.8 uses the complete official Python SDK and a native ARM compiler.
RUN test "$(uname -m)" = aarch64 && apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev
COPY appworld-core-client-py311-observed.txt /trusted/observed-constraints.txt
COPY verify_appworld_core_cpu_environment.py /trusted/verify_environment.py
WORKDIR /work
# This is a previously observed core-client resolution, NOT an author lock.
# No datasets, Agent experiments, models or inference-service packages are added.
RUN /uv venv /opt/venv && \
    /uv pip install --index-url https://pypi.org/simple \
    -r /trusted/observed-constraints.txt && \
    /opt/venv/bin/python -m appworld.cli install && \
    /opt/venv/bin/python -m pip check && \
    /opt/venv/bin/python /trusted/verify_environment.py \
    --lock /trusted/observed-constraints.txt > /trusted/environment-qualification.json
ENV PATH=/opt/venv/bin:$PATH
# Actual task execution must additionally provide isolated writable task state.
# This core image alone is not a qualified Agent run or scoring result.
USER 65534:65534
ENTRYPOINT ["/opt/venv/bin/python", "-m", "appworld.cli"]
CMD ["--help"]
