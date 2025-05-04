import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
import random

class QuantizedModelPipeline:
    def __init__(self,datset_name, model_name, quantization_config=None, max_tokens=64):
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.max_tokens = max_tokens
        self.device = torch.device("cuda:1")
        self.model = None
        self.tokenizer = None
        self.model_size_mb = None  # Track model size
        self.benchmark_dataset_name = datset_name
        self.n_samples_to_benchmark = 1000

    def load_quantized_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.quantization_config is None:
            # Full precision baseline
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda:1",
                torch_dtype=torch.float32
            )
            print(f"Loaded full-precision baseline model: {self.model_name}")
        else:
            # Quantized model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.quantization_config,
                device_map="cuda:1",
                torch_dtype=torch.bfloat16
            )
            print(f"Loaded quantized model: {self.model_name}")

        self.model.eval()
        self.model_size_mb = self._calculate_model_size()
        print(f"Model size after loading: {self.model_size_mb:.2f} MB")

    def _calculate_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 ** 2)  # Convert to MB

    def calculate_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     self.model.resize_token_embeddings(len(self.tokenizer))

        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            end_time = time.time()
        loss = outputs.loss.item()
        return np.exp(loss), end_time - start_time

    def generate_and_evaluate_bleu(self, input_text, reference_text):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        start_time = time.time()
        outputs = self.model.generate(inputs, max_new_tokens=self.max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        end_time = time.time()

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        smoothie = SmoothingFunction().method4
        reference = reference_text
        candidate = generated_text.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

        return generated_text, bleu_score, end_time - start_time

    def evaluate_sample(self, input_text, reference_text):
        ppl, ppl_time = self.calculate_perplexity(input_text)
        generated_text, bleu, gen_time = self.generate_and_evaluate_bleu(input_text, reference_text)
        avg_perplexity = self.benchmark_on_dataset()

        return {
            "Dataset":self.benchmark_dataset_name,
            "generated": generated_text,
            # "perplexity": ppl,
            # "perplexity_time": ppl_time,
            "bleu_score": bleu,
            "generation_time": gen_time,
            "model_size_mb": self.model_size_mb,
            "average_perplexity":avg_perplexity
        }

    def get_transformer_layers(self):
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers
            elif hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
                return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise ValueError("Unsupported model architecture: can't find transformer layers")

    def compute_layerwise_importance(self, text):
        print("\n🔍 Computing layerwise importance (Ablation)...")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        labels = inputs["input_ids"]

        with torch.no_grad():
            full_output = self.model(**inputs, labels=labels)
            base_loss = full_output.loss.item()
            base_perplexity = np.exp(base_loss)

        importance_scores = {}

        layers = self.get_transformer_layers()

        for i, layer in enumerate(layers):
            def forward_hook(module, input, output):
                if isinstance(output, tuple):
                    return tuple(torch.zeros_like(t) if isinstance(t, torch.Tensor) else t for t in output)
                return torch.zeros_like(output) if isinstance(output, torch.Tensor) else output

            handle = layer.register_forward_hook(forward_hook)

            with torch.no_grad():
                ablated_output = self.model(**inputs, labels=labels)
                ablated_loss = ablated_output.loss.item()
                ablated_perplexity = np.exp(ablated_loss)

            delta_loss = ablated_loss - base_loss
            delta_perplexity = ablated_perplexity - base_perplexity

            importance_scores[i] = {
                "delta_loss": delta_loss,
                "delta_perplexity": delta_perplexity,
                "ablated_loss": ablated_loss,
                "ablated_perplexity": ablated_perplexity
            }

            handle.remove()
            print(f"Layer {i}: ΔLoss = {delta_loss:.4f}, ΔPPL = {delta_perplexity:.4f}")

        return importance_scores


    def print_model_layers(self):
        print(f"\n📚 Model Layers in: {self.model_name}")
        print("=" * 50)
        for name, module in self.model.named_modules():
            print(name)
    
    def extract_text(self, sample, dataset_name):
        """Handle different dataset formats and extract text for GPT-2."""
        if dataset_name == "squad":
            return sample["context"]
        elif dataset_name.startswith("glue"):
            # Choose which GLUE subset to use
            text_a = sample.get("sentence") or sample.get("sentence1", "")
            text_b = sample.get("sentence2", "")
            return text_a + (" " + text_b if text_b else "")
        elif dataset_name == "wikitext":
            return sample["text"]
        else:
            # Fallback: try the first string field
            for val in sample.values():
                if isinstance(val, str):
                    return val
        return None
    
    def benchmark_on_dataset(self):
            # Load dataset and subset
            split = "train"
            dataset_name = self.benchmark_dataset_name
            max_samples = self.n_samples_to_benchmark
            subset = None
            if(dataset_name == 'glue'):
                subset = "mrpc"
            if(dataset_name == 'wikitext'):
                subset="wikitext-2-raw-v1"
            
            if subset:
                dataset = load_dataset(dataset_name, subset, split=f"{split}[:{max_samples}]")
            else:
                dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")

            perplexities = []

            for sample in dataset:
                text = self.extract_text(sample, dataset_name)
                if not text or len(text.strip()) < 5:
                    continue

                encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = encodings.input_ids
                if input_ids.shape[1] < 2:
                    continue

                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda:1")

                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)

            if perplexities:
                avg_perplexity = sum(perplexities) / len(perplexities)
                print(f"Average Perplexity on {dataset_name} ({split}): {avg_perplexity:.2f}")
            else:
                print(f"No valid samples found for {dataset_name}.")
            return avg_perplexity

    def evaluate_datset(self):
        QA_pairs = self.get_QA_pairs(self.benchmark_dataset_name)
        random.shuffle(QA_pairs)
        data_size = len(QA_pairs)

        QA_pairs = QA_pairs[:self.n_samples_to_benchmark]
        total_perplexity = 0
        total_bleuScore = 0
        total_gen_time = 0

        for ques,ans in QA_pairs:
            ppl, _ = self.calculate_perplexity(ques)
            _, bleu, gen_time = self.generate_and_evaluate_bleu(ques, ans)
            total_perplexity+=ppl
            total_bleuScore+=bleu
            total_gen_time+=gen_time

        return {
            "Dataset":self.benchmark_dataset_name,
            "model_size_mb": self.model_size_mb,
            "average_bleu_score": total_bleuScore/data_size,
            "average_generation_time": total_gen_time/data_size,
            "average_perplexity":total_perplexity/data_size
        }

    def load_datsets(self, dataset_name):
        dataset = None
        if dataset_name  == 'squad':
            return(load_dataset("squad", split="train"))
        elif dataset_name == 'triviaQa':         
            return(load_dataset("trivia_qa", "rc", split="train"))
        elif dataset_name == 'repliQa':
            return(load_dataset("ServiceNow/repliqa", split="repliqa_0"))
        else:
            print("Invalid Dataset")
            return None

    def get_QA_pairs(self, dataset_name):
        dataset = self.load_datsets(dataset_name)
        QA_pairs=[]
        if dataset_name  == 'squad':
            for item in dataset:
                QA_pairs.append([item['question'], item['answers']['text'][0]])
        elif dataset_name == 'repliQa':
            for item in dataset:
                QA_pairs.append([item['question'], item['answer']])
        elif dataset_name == 'triviaQa': 
            for item in dataset:
                QA_pairs.append([item['question'], item["answer"]["value"]])
        return QA_pairs

        

