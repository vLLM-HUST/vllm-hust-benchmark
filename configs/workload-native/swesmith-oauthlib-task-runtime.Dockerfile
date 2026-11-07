# Task runtime extends the qualified ARM base, preserving original Git objects.
# Pass a base image whose identity matches the recorded compatibility receipt.
ARG BASE_IMAGE=workload-swesmith-oauthlib-arm:20261105-001
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY repository/ /testbed/
WORKDIR /testbed
