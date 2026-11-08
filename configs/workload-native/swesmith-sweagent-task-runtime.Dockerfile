# Native remote execution/tool packages staged before sandbox execution.
ARG BASE_IMAGE=workload-native/swesmith.arm64.oauthlib_1776_oauthlib.1fd52536:latest
FROM ${BASE_IMAGE}
COPY original-requirements.txt /opt/original-requirements.txt
RUN python -m pip install --no-cache-dir --no-compile --only-binary=tree-sitter,tree-sitter-languages \
    'swe-rex==1.4.0' 'tree-sitter==0.21.3' 'tree-sitter-languages==1.10.2'
RUN python -m pip check
RUN python -c "import importlib.metadata as m; from packaging.requirements import Requirement; rows=[Requirement(s) for s in open('/opt/original-requirements.txt') if s.strip() and not s.startswith('#')]; assert all(r.specifier.contains(m.version(r.name)) for r in rows)"
