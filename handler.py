import logging
import os

import runpod
import torch

import windowseat
from windowseat import sync
from windowseat.inference import run_inference
from windowseat.network import load_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Get input/output directories and create if missing
INPUT_DIR = os.path.join(CUR_DIR, "inputs")
os.makedirs(INPUT_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(CUR_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info(
    "Loading network (model=%s, lora=%s, device=%s)",
    windowseat.MODEL,
    windowseat.LORA_MODEL_URI,
    windowseat.DEVICE,
)
vae, transformer, embeds_dict, processing_resolution = load_network(
    windowseat.MODEL, windowseat.LORA_MODEL_URI, torch.device(windowseat.DEVICE)
)
logger.info(
    "Network loaded successfully (processing_resolution=%s)", processing_resolution
)


def handler(job):
    logger.info("Job received: %s", job.get("id"))

    # Extract input from the job
    job_input = job["input"]
    imageid = job_input.get("imageid")

    # Validate input
    if not imageid:
        logger.error("Job %s missing required field 'imageid'", job.get("id"))
        return {"error": "No image provided for analysis."}

    try:
        logger.info("Downloading image: imageid=%s", imageid)
        sync.download(INPUT_DIR, [imageid])
        logger.info("Download complete: imageid=%s", imageid)

        logger.info("Starting inference: imageid=%s", imageid)
        run_inference(
            vae,
            transformer,
            embeds_dict,
            processing_resolution,
            INPUT_DIR,
            OUTPUT_DIR,
            False,
        )
        logger.info("Inference complete: imageid=%s", imageid)

        logger.info("Uploading results: imageid=%s", imageid)
        sync.upload(OUTPUT_DIR)
        logger.info("Upload complete: imageid=%s", imageid)
    except Exception as e:
        logger.exception("Job %s failed: %s", job.get("id"), e)
        return {"error": str(e)}

    return {"status": "complete"}


runpod.serverless.start({"handler": handler})
