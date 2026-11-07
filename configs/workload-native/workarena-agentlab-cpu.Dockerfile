# Build the existing WebArena/AgentLab CPU client first. The external context
# must contain only verified original WorkArena source in an author/ directory.
ARG BASE_CLIENT=local/webarena-client:0.4.2
FROM ${BASE_CLIENT}
COPY author /opt/workarena-author
RUN python -m pip freeze --all | grep -vi '^browsergym-workarena==' > /tmp/base-constraints.txt \
    && python -m pip install --no-cache-dir -c /tmp/base-constraints.txt /opt/workarena-author \
    && python -m pip check
