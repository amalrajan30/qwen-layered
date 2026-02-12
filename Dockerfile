# --------------------------------------------------------------------------
# RunPod Serverless Worker — Qwen-Image-Layered
#
# Slim image. Model weights are provided by RunPod's cached model feature:
#   Set "Qwen/Qwen-Image-Layered" in the Model field when creating the
#   endpoint.  RunPod pre-loads the weights on the host at
#   /runpod-volume/huggingface-cache/hub/ so workers start fast.
# --------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt

COPY src/ /src/

CMD ["python", "/src/handler.py"]
