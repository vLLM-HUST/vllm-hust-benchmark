# Dependency-only extension; build on the qualified public browser image.
ARG RENDER_BASE
FROM ${RENDER_BASE}
USER 0:0
RUN /uv pip install --python /opt/venv/bin/python --index-url https://pypi.org/simple Markdown==3.11.2 && \
    /uv pip check --python /opt/venv/bin/python && \
    /opt/venv/bin/python -c "from importlib.metadata import version; import markdown; assert version('Markdown') == '3.11.2'; assert markdown.markdown('fixture') == '<p>fixture</p>'"
USER 65534:65534
