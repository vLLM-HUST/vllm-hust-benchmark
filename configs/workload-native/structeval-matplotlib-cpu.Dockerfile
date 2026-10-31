# Minimal renderer component only. No AppWorld/protected code, tasks or models.
FROM ghcr.io/astral-sh/uv@sha256:43b6b8ab14b48e7850828739e984661755f25cb48b5f3a812d846df8846ff5cf AS uv
FROM public.ecr.aws/docker/library/python@sha256:8f51ccb8569b75cf967ffe3e1638b9953042bf36142d7032cbb6f0696c9cd2c6
COPY --from=uv /uv /uv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    MPLCONFIGDIR=/tmp/mpl-cache MPLBACKEND=Agg
COPY structeval-matplotlib-py311-observed.txt /trusted/observed.txt
COPY verify_structeval_matplotlib_environment.py /trusted/verify_environment.py
RUN test "$(uname -m)" = aarch64 && /uv venv /opt/venv && \
    /uv pip install --python /opt/venv/bin/python --index-url https://pypi.org/simple \
      -r /trusted/observed.txt && \
    /uv pip check --python /opt/venv/bin/python && \
    /opt/venv/bin/python /trusted/verify_environment.py --lock /trusted/observed.txt \
      > /trusted/environment-qualification.json
USER 65534:65534
# Task execution additionally requires network-none, read-only root, bounded
# writable tmp/output, no capabilities/devices and original verified source.
ENTRYPOINT ["/opt/venv/bin/python", "/trusted/verify_environment.py"]
CMD ["--lock", "/trusted/observed.txt"]
