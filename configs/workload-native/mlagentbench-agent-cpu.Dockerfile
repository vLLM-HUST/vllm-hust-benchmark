# Explicitly extend the qualified CIFAR task image, never a shared host runtime.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY wheels/ /opt/mlagentbench-agent/wheels/
COPY requirements-verified.txt /opt/mlagentbench-agent/requirements-verified.txt
RUN python -m pip install --no-index --no-deps --require-hashes --find-links /opt/mlagentbench-agent/wheels -r /opt/mlagentbench-agent/requirements-verified.txt
RUN python -c "import anthropic, dacite; assert anthropic.HUMAN_PROMPT == '\n\nHuman:'; assert anthropic.AI_PROMPT == '\n\nAssistant:'"
