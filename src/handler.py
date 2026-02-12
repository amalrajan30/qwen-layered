"""
RunPod Serverless handler for Qwen-Image-Layered.

Decomposes an input image into multiple RGBA layers using the
QwenImageLayeredPipeline from Hugging Face diffusers.

Model is loaded from RunPod's model cache at startup.  The snapshot
path is resolved from /runpod-volume/huggingface-cache/hub/ and passed
directly to from_pretrained with local_files_only=True.

Reference: https://github.com/runpod-workers/model-store-cache-example
"""

import base64
import io
import os
import time
import traceback

import runpod
import torch
from PIL import Image

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen-Image-Layered")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Force offline mode so we never silently download 57 GB
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

pipeline = None


def resolve_snapshot_path(model_id: str) -> str:
    """
    Resolve the local snapshot path for a RunPod-cached model.

    Follows the HuggingFace cache layout:
      hub/models--{org}--{name}/refs/main   → commit hash
      hub/models--{org}--{name}/snapshots/{hash}/  → model files
    """
    if "/" not in model_id:
        raise ValueError(f"MODEL_NAME '{model_id}' is not in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    print(f"[ModelStore] MODEL_NAME : {model_id}")
    print(f"[ModelStore] Model root : {model_root}")

    # Try to read the snapshot hash from refs/main
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            print(f"[ModelStore] Using snapshot from refs/main: {candidate}")
            return candidate

    # Fall back to first available snapshot
    if not os.path.isdir(snapshots_dir):
        raise RuntimeError(
            f"[ModelStore] snapshots directory not found: {snapshots_dir}. "
            "Ensure 'Qwen/Qwen-Image-Layered' is set in the Model field "
            "of your RunPod endpoint configuration."
        )

    versions = [
        d
        for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    if not versions:
        raise RuntimeError(
            f"[ModelStore] No snapshot subdirectories under {snapshots_dir}"
        )

    versions.sort()
    chosen = os.path.join(snapshots_dir, versions[0])
    print(f"[ModelStore] Using first available snapshot: {chosen}")
    return chosen


def load_model():
    """Load the Qwen-Image-Layered pipeline once at worker startup."""
    global pipeline

    from diffusers import QwenImageLayeredPipeline

    local_path = resolve_snapshot_path(MODEL_NAME)
    print(f"[ModelStore] Resolved local path: {local_path}")

    start = time.time()

    pipeline = QwenImageLayeredPipeline.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    elapsed = time.time() - start
    print(f"[ModelStore] Pipeline loaded in {elapsed:.1f}s")


def decode_image(image_input: str) -> Image.Image:
    """Decode a base64-encoded image string into a PIL RGBA Image."""
    image_bytes = base64.b64decode(image_input)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    return image


def encode_image(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    """
    RunPod handler for image layer decomposition.

    Expected input:
    {
        "image": "<base64-encoded image>",
        "layers": 4,                   # optional, default 4
        "resolution": 640,             # optional, 640 or 1024
        "num_inference_steps": 50,      # optional, default 50
        "true_cfg_scale": 4.0,          # optional, default 4.0
        "seed": 777,                    # optional, for reproducibility
        "negative_prompt": " ",         # optional
        "cfg_normalize": true,          # optional, default true
        "use_en_prompt": true           # optional, default true
    }

    Returns:
    {
        "layers": ["<base64 PNG>", ...],
        "num_layers": int,
        "inference_time_seconds": float
    }
    """
    job_input = job["input"]

    # --- Validate required fields ---
    if "image" not in job_input:
        return {"error": "Missing required field: 'image' (base64-encoded image)"}

    try:
        image = decode_image(job_input["image"])
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}

    # --- Extract optional parameters ---
    layers = job_input.get("layers", 4)
    resolution = job_input.get("resolution", 640)
    num_inference_steps = job_input.get("num_inference_steps", 50)
    true_cfg_scale = job_input.get("true_cfg_scale", 4.0)
    seed = job_input.get("seed", None)
    negative_prompt = job_input.get("negative_prompt", " ")
    cfg_normalize = job_input.get("cfg_normalize", True)
    use_en_prompt = job_input.get("use_en_prompt", True)

    # --- Build inference kwargs ---
    inference_kwargs = {
        "image": image,
        "layers": layers,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": 1,
        "cfg_normalize": cfg_normalize,
        "use_en_prompt": use_en_prompt,
    }

    if seed is not None:
        inference_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(
            int(seed)
        )

    # --- Run inference ---
    try:
        start = time.time()

        with torch.inference_mode():
            output = pipeline(**inference_kwargs)

        layer_images = output.images[0]  # List of PIL RGBA images
        inference_time = time.time() - start

        # Encode each layer as base64 PNG
        encoded_layers = [encode_image(layer) for layer in layer_images]

        return {
            "layers": encoded_layers,
            "num_layers": len(encoded_layers),
            "inference_time_seconds": round(inference_time, 2),
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "error": "CUDA out of memory. Try reducing resolution (640) or layers count."
        }
    except Exception as e:
        print(f"Inference error: {traceback.format_exc()}")
        return {"error": f"Inference failed: {str(e)}"}


# Load model at worker startup (before processing any jobs)
load_model()

runpod.serverless.start({"handler": handler})
