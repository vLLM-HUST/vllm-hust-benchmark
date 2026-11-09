# Explicit qualified Python3.10/aarch64 scientific CPU Agent base.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY software/ /opt/mlagentbench-graph/software/
COPY requirements-*.txt /opt/mlagentbench-graph/
RUN MAX_JOBS=1 FORCE_CUDA=0 python -m pip install --no-index --no-deps --no-build-isolation --require-hashes --find-links /opt/mlagentbench-graph/software -r /opt/mlagentbench-graph/requirements-core.txt
RUN MAX_JOBS=1 FORCE_CUDA=0 python -m pip install --no-index --no-deps --no-build-isolation --require-hashes --find-links /opt/mlagentbench-graph/software -r /opt/mlagentbench-graph/requirements-scatter.txt
RUN MAX_JOBS=1 FORCE_CUDA=0 python -m pip install --no-index --no-deps --no-build-isolation --require-hashes --find-links /opt/mlagentbench-graph/software -r /opt/mlagentbench-graph/requirements-sparse.txt
# Keep scientific libraries within the bounded task/grader PID budget.
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
