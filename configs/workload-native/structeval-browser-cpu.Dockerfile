# Build on your qualified minimal Matplotlib image; software only, no tasks.
ARG RENDER_BASE
FROM ${RENDER_BASE}
USER 0:0
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
COPY structeval-browser-py311-observed.txt /trusted/browser-observed.txt
COPY verify_structeval_browser_environment.py /trusted/verify_browser_environment.py
RUN /uv pip install --python /opt/venv/bin/python --index-url https://pypi.org/simple \
      -r /trusted/browser-observed.txt && \
    /uv pip check --python /opt/venv/bin/python && \
    DEBIAN_FRONTEND=noninteractive /opt/venv/bin/python -m playwright install --with-deps chromium && \
    /opt/venv/bin/python /trusted/verify_browser_environment.py --lock /trusted/browser-observed.txt \
      > /trusted/browser-environment.json
USER 65534:65534
ENTRYPOINT ["/opt/venv/bin/python", "/trusted/verify_browser_environment.py"]
CMD ["--lock", "/trusted/browser-observed.txt"]
