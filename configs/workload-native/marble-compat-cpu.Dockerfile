# Software-only import/runtime base, not complete per-scenario environments.
# Caller must verify the explicit local Ubuntu22.04 image ID before building.
ARG MARBLE_BASE_IMAGE
FROM ${MARBLE_BASE_IMAGE}
USER 0:0
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 libpython3.10-stdlib libgomp1 ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
# Context contains only the separately audited Python3.10 environment (93+2
# packages) and99 original author files with the approved3-line evaluator fix.
# No tasks, answers, models, HF cache, outputs or credentials in the context.
COPY client/ /opt/marble-client/
COPY author/ /author/
RUN /opt/marble-client/bin/python -I -c "import sys; assert sys.version_info[:2] == (3,10)" \
    && mkdir /work /offline-model-cache \
    && chown 65534:65534 /work /offline-model-cache
ENV PYTHONPATH="" \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/offline-model-cache \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /work
USER 65534:65534
# Actual tasks additionally require readonly-root/network-none/resource limits,
# explicit private data/output deployment, local model transport and judges.
ENTRYPOINT ["/opt/marble-client/bin/python", "-I", "-c", "import runpy,sys; sys.path.insert(0,'/author'); sys.argv=['marble.main',*sys.argv[1:]]; runpy.run_module('marble.main',run_name='__main__')"]
CMD ["--help"]
