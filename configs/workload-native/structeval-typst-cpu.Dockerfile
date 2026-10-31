# Build on your qualified minimal Matplotlib image; this is software only.
# Original author requirements do not pin Typst/ImageMagick. Versions below are
# an explicitly disclosed environment choice, not an original author lock.
ARG RENDER_BASE
FROM ${RENDER_BASE}
USER 0:0
RUN apt-get update && apt-get install -y --no-install-recommends \
      imagemagick ghostscript fontconfig fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
# Obtain only from https://github.com/typst/typst/releases/tag/v0.13.1.
# No binary is redistributed in this repository. This is an observed checksum,
# not a claim that GitHub published a vendor checksum for this old asset.
COPY typst /usr/local/bin/typst
COPY verify_structeval_typst_environment.py /trusted/verify_typst_environment.py
RUN /opt/venv/bin/python -c "import hashlib; from pathlib import Path; p=Path('/usr/local/bin/typst'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'ce3c5ddac1e47bd75d9a8cc0c7620158873e70719fdfc1673beb1da474ab2e95'; p.chmod(0o555)" && \
    /opt/venv/bin/python /trusted/verify_typst_environment.py --prepare-pdf-read-policy
RUN /opt/venv/bin/python /trusted/verify_typst_environment.py > /trusted/typst-environment.json
USER 65534:65534
ENTRYPOINT ["/opt/venv/bin/python", "/trusted/verify_typst_environment.py"]
CMD []
