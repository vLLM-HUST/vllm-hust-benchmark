FROM naturebench-base:v3
LABEL workload.owner=naturebench-native-20261102
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends libboost-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev libboost-program-options-dev libcgal-dev libeigen3-dev libtbb-dev
RUN python -m pip install cython pybind11
ADD gudhi.3.10.1.tar.gz /opt/gudhi-source/
RUN cmake -S /opt/gudhi-source/gudhi.3.10.1 -B /opt/gudhi-build -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/py311/bin/python -DWITH_GUDHI_TEST=OFF -DWITH_GUDHI_EXAMPLE=OFF -DWITH_GUDHI_UTILITIES=OFF -DWITH_GUDHI_GUDHUI=OFF
RUN cd /opt/gudhi-build/python && python -m pip wheel --no-build-isolation --no-deps . -w /opt/gudhi-wheels
RUN python -m pip install --no-deps /opt/gudhi-wheels/*.whl && python -c "import gudhi; assert gudhi.__version__ == '3.10.1'; print(gudhi.__debug_info__); print(gudhi.__available_modules); print(gudhi.__missing_modules)"
