import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset




class QuantizedModelPipeline:
    def __init__(self, model_name, quantization_config=None, max_tokens=64):
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_quantized_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        print(f"Loaded model: {self.model_name} with 4-bit quantization")

    def calculate_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            end_time = time.time()
        loss = outputs.loss.item()
        return np.exp(loss), end_time - start_time

    def generate_and_evaluate_bleu(self, input_text, reference_text):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        start_time = time.time()
        outputs = self.model.generate(inputs, max_new_tokens=self.max_tokens)
        end_time = time.time()

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        smoothie = SmoothingFunction().method4
        reference = [reference_text.split()]
        candidate = generated_text.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

        return generated_text, bleu_score, end_time - start_time

    def evaluate_sample(self, input_text, reference_text):
        ppl, ppl_time = self.calculate_perplexity(input_text)
        generated_text, bleu, gen_time = self.generate_and_evaluate_bleu(input_text, reference_text)

        return {
            "generated": generated_text,
            "perplexity": ppl,
            "perplexity_time": ppl_time,
            "bleu_score": bleu,
            "generation_time": gen_time
        }
