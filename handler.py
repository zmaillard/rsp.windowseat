import os

import runpod
import torch

import windowseat
from windowseat import sync
from windowseat.inference import run_inference
from windowseat.network import load_network

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Get input/output directories and create if missing
INPUT_DIR = os.path.join(CUR_DIR, "inputs")
os.makedirs(INPUT_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(CUR_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


vae, transformer, embeds_dict, processing_resolution = load_network(
    windowseat.MODEL, windowseat.LORA_MODEL_URI, torch.device(windowseat.DEVICE)
)


def handler(job):
    # Extract input from the job
    job_input = job["input"]
    imageid = job_input.get("imageid")

    # Validate input
    if not imageid:
        return {"error": "No image provided for analysis."}

    sync.download(INPUT_DIR, [imageid])

    run_inference(
        vae,
        transformer,
        embeds_dict,
        processing_resolution,
        INPUT_DIR,
        OUTPUT_DIR,
        False,
    )

    sync.upload(OUTPUT_DIR)


runpod.serverless.start({"handler": handler})
