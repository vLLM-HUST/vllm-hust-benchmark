# CPU software only; official scorer and private databases stay outside Git.
# The author requires sqlparse/nltk without versions. These are disclosed
# integration pins, not an author-supplied lock or native benchmark result.
FROM ghcr.io/astral-sh/uv@sha256:43b6b8ab14b48e7850828739e984661755f25cb48b5f3a812d846df8846ff5cf AS uv
FROM public.ecr.aws/docker/library/python@sha256:8f51ccb8569b75cf967ffe3e1638b9953042bf36142d7032cbb6f0696c9cd2c6
COPY --from=uv /uv /uv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    NLTK_DATA=/opt/nltk_data PYTHONDONTWRITEBYTECODE=1
COPY verify_spider_test_suite_environment.py /trusted/verify_environment.py
# External build-context asset, never committed. Source: nltk/nltk_data
# @550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip.
COPY punkt_tab.zip /trusted/punkt_tab.zip
RUN /uv venv /opt/venv && \
    /uv pip install --python /opt/venv/bin/python --index-url https://pypi.org/simple \
      sqlparse==0.5.3 nltk==3.9.2 tqdm==4.67.1 \
      click==8.5.0 cloudpickle==3.1.2 joblib==1.6.0 regex==2026.9.29 && \
    /uv pip check --python /opt/venv/bin/python && \
    /opt/venv/bin/python -c "import hashlib,zipfile; from pathlib import Path; p=Path('/trusted/punkt_tab.zip'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106'; zipfile.ZipFile(p).extractall('/opt/nltk_data/tokenizers')" && \
    /opt/venv/bin/python /trusted/verify_environment.py > /trusted/environment-qualification.json
USER 65534:65534
# Native SQL execution requires an independent fresh nonroot, read-only,
# network-none container with bounded resources and no host/device/socket access.
ENTRYPOINT ["/opt/venv/bin/python", "/trusted/verify_environment.py"]
CMD []
