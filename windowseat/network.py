import json
import logging

import safetensors
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig

logger = logging.getLogger(__name__)


def fetch_state_dict(
    pretrained_model_name_or_path_or_dict: str,
    weight_name: str,
    use_safetensors: bool = True,
    subfolder: str | None = None,
):
    logger.debug(
        "Fetching state dict: repo=%s weight=%s subfolder=%s",
        pretrained_model_name_or_path_or_dict,
        weight_name,
        subfolder,
    )
    file_path = hf_hub_download(
        pretrained_model_name_or_path_or_dict, weight_name, subfolder=subfolder
    )
    logger.debug("State dict downloaded to: %s", file_path)
    if use_safetensors:
        state_dict = safetensors.torch.load_file(file_path)
    else:
        state_dict = torch.load(file_path, weights_only=True)
    return state_dict


def load_qwen_transformer(uri: str, device: torch.device):
    logger.info("Loading transformer: uri=%s device=%s", uri, device)
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )

    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=nf4,
        device_map=device,
    )

    logger.info("Transformer loaded successfully")
    return transformer


def load_lora_into_transformer(uri: str, transformer: QwenImageTransformer2DModel):
    logger.info("Loading LoRA adapter: uri=%s", uri)
    lora_config = LoraConfig.from_pretrained(uri, subfolder="transformer_lora")
    transformer.add_adapter(lora_config)
    state_dict = fetch_state_dict(
        uri, "pytorch_lora_weights.safetensors", subfolder="transformer_lora"
    )
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    logger.info(
        "LoRA state dict loaded: missing_keys=%d unexpected_keys=%d",
        len(missing),
        len(unexpected),
    )
    if len(unexpected) > 0:
        logger.error("Unexpected keys in transformer state dict: %s", unexpected)
        raise ValueError(f"Unexpected keys in transformer state dict: {unexpected}")
    logger.info("LoRA adapter applied successfully")
    return transformer


def load_qwen_vae(uri: str, device: torch.device):
    logger.info("Loading VAE: uri=%s device=%s", uri, device)
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    logger.info("VAE loaded successfully")
    return vae


def load_network(uri_base: str, uri_lora: str, device: torch.device):
    logger.info(
        "Loading full network: uri_base=%s uri_lora=%s device=%s",
        uri_base,
        uri_lora,
        device,
    )
    config_file = hf_hub_download(uri_lora, "model_index.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    processing_resolution = config_dict["processing_resolution"]
    logger.info("Config loaded: processing_resolution=%s", processing_resolution)

    vae = load_qwen_vae(uri_base, device)
    transformer = load_qwen_transformer(uri_base, device)
    load_lora_into_transformer(uri_lora, transformer)
    embeds_dict = load_embeds_dict(uri_lora)
    logger.info("Full network loaded successfully")
    return vae, transformer, embeds_dict, processing_resolution


def load_embeds_dict(uri: str):
    logger.info("Loading text embeddings: uri=%s", uri)
    embeds_dict = fetch_state_dict(
        uri, "state_dict.safetensors", subfolder="text_embeddings"
    )
    logger.info("Text embeddings loaded: keys=%d", len(embeds_dict))
    return embeds_dict
