# --------------------------------------------------------------------------
# RunPod Serverless Worker — Qwen-Image-Layered
#
# Slim image. Model weights are provided by RunPod's cached model feature:
#   Set "Qwen/Qwen-Image-Layered" in the Model field when creating the
#   endpoint.  RunPod pre-loads the weights on the host at
#   /runpod-volume/huggingface-cache/hub/ before the worker starts.
#
# Reference: https://github.com/runpod-workers/model-store-cache-example
# --------------------------------------------------------------------------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/handler.py .

CMD ["python", "-u", "handler.py"]
