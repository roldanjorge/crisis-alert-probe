# %%
import torch as t
from transformer_lens import HookedTransformer
import os
from pathlib import Path
from transformer_lens.utils import get_device
import circuitsvis as cv
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")


def get_model():
    """
    Get the model Llama-2-13b-chat-hf from Hugging Face
    """
    device = get_device()
    print(f"Using device: {device}")

    model_name = "meta-llama/Llama-2-13b-chat-hf"

    try:
        print(f"Loading model: {model_name}")
        print("This will download from Hugging Face if not cached locally...")
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        print(f"Model loaded successfully on {device}")
        print(f"Model config: {model.cfg}")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you're logged into Hugging Face: huggingface-cli login")
