# Software only, not an original-author dependency lock. Use a qualified
# minimal Matplotlib base. Installers may use a separately qualified build
# network; generated TeX must still run in the bounded --network none worker.
ARG RENDER_BASE
FROM ${RENDER_BASE}
USER 0:0
RUN apt-get update && apt-get install -y --no-install-recommends \
      texlive-latex-extra texlive-luatex texlive-fonts-recommended \
      texlive-pictures poppler-utils fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
RUN /uv pip install --python /opt/venv/bin/python --index-url https://pypi.org/simple pdf2image==1.17.0 && \
    /uv pip check --python /opt/venv/bin/python
ENV TEXMFVAR=/tmp/texmf-var TEXMFCACHE=/tmp/texmf-cache
COPY verify_structeval_latex_environment.py /trusted/verify_latex_environment.py
RUN /opt/venv/bin/python /trusted/verify_latex_environment.py > /trusted/latex-environment.json
USER 65534:65534
ENTRYPOINT ["/opt/venv/bin/python", "/trusted/verify_latex_environment.py"]
CMD []
