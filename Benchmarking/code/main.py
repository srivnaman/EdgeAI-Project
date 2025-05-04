import torch
from transformers import BitsAndBytesConfig
import os
import warnings
from evaluate import evaluate_
import argparse
import config

warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

def parse_args():
    parser = argparse.ArgumentParser(description='LLM Benchmarking')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='squad')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # MODEL_NAMES ['gpt2', 'facebook/opt-350m', 'distilgpt2']
    
    # Override config dynamically
    config.MODEL_NAME = args.model_name
    config.DATASET = args.dataset
    # Different quantization settings  Uncomment the other configs when using gpu
    experiments = {
        # "baseline": None,  # Full precision
        "8bit": BitsAndBytesConfig(load_in_8bit=True),
        # "4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    }

    for exp_name, exp_config in experiments.items():
        evaluate_(exp_name, exp_config)


if __name__ == "__main__":
    main()
