#import huggingface_hub
#token = 
#huggingface_hub.login(token=token)

import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

APPLICATION_NAME = __name__
logger = logging.getLogger(APPLICATION_NAME)
logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_DIR = Path("assets_llmn")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def save_model(device, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    logger.info(f"Using device {device} to save model to {MODEL_DIR}")

    # use 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    logger.info("Downloading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Saving model to {MODEL_DIR}")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    logger.info("Model and tokenizer saved")


if not (MODEL_DIR / "config.json").exists():
    model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    logger.info(f"Downloading model {model_name}")
    save_model(DEVICE, model_name)
else:
    logger.info("Using existing local model")
