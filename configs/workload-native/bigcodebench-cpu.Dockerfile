# Fixed official Python image via the public Docker-library registry mirror.
FROM public.ecr.aws/docker/library/python@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN test "$(uname -m)" = aarch64 && apt-get update && \
    apt-get install -y --no-install-recommends git g++ python3-tk zip unzip \
    procps r-base libgdal-dev libfreetype6-dev libpng-dev pkg-config \
    python3-dev python3-matplotlib libsndfile1
COPY author/ /author/
COPY requirements.lock.txt /trusted/requirements.lock.txt
COPY audit_bigcodebench_scoring_assets.py run_bigcodebench_original_generation.py \
    bigcodebench_container_worker.py verify_bigcodebench_cpu_environment.py /trusted/
RUN OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pip install --index-url https://pypi.org/simple \
    numpy==1.21.2 scipy==1.7.2 Cython==0.29.37 setuptools==68.2.2 wheel==0.41.3 && \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pip install --index-url https://pypi.org/simple \
    --no-build-isolation --no-deps scikit-image==0.18.0 && \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pip install --index-url https://pypi.org/simple \
    -c /author/Requirements/requirements-eval.txt -r /trusted/requirements.lock.txt
# Use the unchanged source tree and its native local-dev version fallback;
# do not synthesize a Git revision or install an unrelated released package.
ENV PYTHONPATH=/author:/trusted HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
RUN python /trusted/verify_bigcodebench_cpu_environment.py --author-root /author \
    --lock /trusted/requirements.lock.txt > /trusted/environment-qualification.json
# Actual scoring is always launched separately with the public nonroot/offline
# isolation controls. No dataset, completion or generated code is in this image.
