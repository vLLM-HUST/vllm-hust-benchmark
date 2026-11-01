# Build the separately qualified core recipe first and supply its actual image ID.
# Context contains original public software only, never task data or model files.
ARG APPWORLD_CORE_IMAGE
FROM ${APPWORLD_CORE_IMAGE}
USER 0:0
RUN apt-get update && apt-get install -y --no-install-recommends g++ make
COPY author/ /author/
COPY litellm-original/ /litellm-original/
# The removed PyPI release is recovered from original v1.37.19 publisher source,
# not replaced with a newer version. Verify the archive before preparing context.
RUN /opt/venv/bin/python -c "import pathlib,tomllib; p=tomllib.loads(pathlib.Path('/litellm-original/pyproject.toml').read_text()); assert p['tool']['poetry']['version']=='1.37.19'" && \
    /uv pip install --index-url https://pypi.org/simple \
      -c /trusted/observed-constraints.txt \
      'litellm @ file:///litellm-original' \
      'appworld[experiments]==0.1.3.post1' requests-unixsocket==0.4.1 && \
    /opt/venv/bin/python -m pip check && \
    /uv pip freeze > /trusted/experiments-observed.txt
ENV PYTHONPATH=/author GRADIO_ANALYTICS_ENABLED=False DO_NOT_TRACK=1
# The original framework uses this default user's cache. No global HOME change.
RUN mkdir -p /nonexistent && chown 65534:65534 /nonexistent
USER 65534:65534
# Execution additionally requires a dedicated writable /nonexistent cache,
# network=none, read-only root/data, no capabilities and separate environment.
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["--version"]
