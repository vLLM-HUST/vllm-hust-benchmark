# Isolated compatibility candidate, not the author's unmodified x86 image.
# Context must contain verified original source, requirements and built wheels.
ARG BASE_IMAGE=python@sha256:97ff6fda70178dee6c144d41030fb88b6ec86d75e1c517fe96b8f62094ea7ac2
FROM ${BASE_IMAGE}
COPY wheels/ /opt/wheels/
COPY requirements.txt /opt/requirements.txt
RUN python -m pip install --no-index --find-links /opt/wheels -r /opt/requirements.txt
COPY source/ /testbed/
WORKDIR /testbed
RUN python -m pip install --no-index --no-deps --no-build-isolation -e .
ENV PYTHONDONTWRITEBYTECODE=1
