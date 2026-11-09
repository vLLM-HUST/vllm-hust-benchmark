# Original fixed SWE-smith software with its declared minimal validation extra.
# This client is trusted orchestration software, never a generated-code sandbox.
ARG BASE_IMAGE=python@sha256:97ff6fda70178dee6c144d41030fb88b6ec86d75e1c517fe96b8f62094ea7ac2
FROM ${BASE_IMAGE}
COPY author.tar.gz /opt/swesmith-author.tar.gz
# The author's KEY_PREDICTION import is present in upstream SWE-bench v4.1.0;
# current v5.0.2 removed it. Preserve the author scorer, pin its dependency.
RUN python -m pip install --no-cache-dir --no-compile 'swebench==4.1.0' '/opt/swesmith-author.tar.gz[validate]'
ENV PYTHONDONTWRITEBYTECODE=1
