# Linux x86_64 Python 3.12 base pinned to an immutable manifest.
FROM python:3.12.12-slim-bookworm@sha256:2986c55feb36e6cae00fa1fefb454283e4b33f35e75ff8bdd123b134130be301
ARG ALD_BACKEND=cu130
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    WANDB_MODE=disabled \
    HF_HUB_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/ald-yolo \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ald
COPY environments/ ./environments/
COPY scripts/install_environment.py scripts/setup_sam3.py ./scripts/
COPY patches/ ./patches/
RUN python scripts/install_environment.py --backend "$ALD_BACKEND" --venv /opt/venv
COPY pipeline/ ./pipeline/
COPY experiment_files/sam3_train_config/ ./experiment_files/sam3_train_config/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY LICENSES/ ./LICENSES/
COPY THIRD_PARTY_NOTICES.md README.md ./
# Run with --user "$(id -u):$(id -g)" so bind-mounted outputs belong to the caller.
USER 10001:10001
ENTRYPOINT ["python", "scripts/run_ald.py"]
CMD ["--help"]
