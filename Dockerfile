FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[p2p]'
COPY examples/workload.py ./workload.py
# No ENTRYPOINT: HF Jobs supplies the driver command.
