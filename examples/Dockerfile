# Supply a tested, immutable vLLM image tag/digest when building.
ARG VLLM_IMAGE
FROM ${VLLM_IMAGE}
USER root
WORKDIR /opt/hfdask
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples/offline.py ./examples/offline.py
RUN pip install --no-cache-dir '.[p2p]'
ENV PYTHONPATH=/opt/hfdask/examples
# vLLM creates subprocesses; Dask workers must not be daemon processes.
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV HF_HUB_OFFLINE=1
ENTRYPOINT []
