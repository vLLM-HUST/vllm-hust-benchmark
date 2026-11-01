# Supply an already qualified Python3.10.22/aarch64 CPU base with the versions
# declared in the task asset profile. This extends it, never mutates that base.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY profile.json /opt/mlagentbench/profile.json
COPY check_mlagentbench_cifar_cpu.py /opt/mlagentbench/check_mlagentbench_cifar_cpu.py
COPY torchvision-requirements.txt /opt/mlagentbench/torchvision-requirements.txt
COPY wheels/ /opt/mlagentbench/wheels/
RUN python -m pip install --no-index --no-deps --require-hashes --find-links /opt/mlagentbench/wheels -r /opt/mlagentbench/torchvision-requirements.txt
RUN python /opt/mlagentbench/check_mlagentbench_cifar_cpu.py --profile /opt/mlagentbench/profile.json --execute
WORKDIR /workspace
