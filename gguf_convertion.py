

# Step 1: Install Dependencies
!apt-get -qq install -y cmake build-essential
!pip install -q transformers sentencepiece

# Step 2: Clone llama.cpp
!git clone https://github.com/ggerganov/llama.cpp.git
# %cd llama.cpp
!make -j

# Step 3: Download GPT-2 (Small) from Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM
import os



# Commented out IPython magic to ensure Python compatibility.

model_name = "gpt2"
save_path = "/Models/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
# %cd /Models/llama.cpp
!python3 convert_hf_to_gguf.py /Models/gpt2 --outfile /Models/gpt2.gguf --outtype q8_0

# Commented out IPython magic to ensure Python compatibility.
# %cd /Models/llama.cpp
!python3 convert_hf_to_gguf.py /Models/gpt2 --outfile /Models/gpt2.gguf --outtype q8_0
## this workde

# Commented out IPython magic to ensure Python compatibility.
# Step 1: Install Dependencies
!apt-get -qq install -y cmake build-essential
!pip install -q transformers sentencepiece

# Step 2: Clone llama.cpp (if not already cloned)
!git clone https://github.com/ggerganov/llama.cpp.git
# %cd llama.cpp
!make -j

# Step 3: Download Models from Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_and_save(model_name, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    del tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(save_path)
    del model

# Model paths
base_dir = "/Models"

# 3.1 Mistral-7B (e.g., "mistralai/Mistral-7B-v0.1")
mistral_model = "mistralai/Mistral-7B-v0.1"
mistral_path = os.path.join(base_dir, "mistral")
download_and_save(mistral_model, mistral_path)

# 3.2 DeepSeek-7B (e.g., "deepseek-ai/deepseek-llm-7b-base")
deepseek_model = "deepseek-ai/deepseek-llm-7b-base"
deepseek_path = os.path.join(base_dir, "deepseek")
download_and_save(deepseek_model, deepseek_path)

# 3.3 LLaMA-2-7B (e.g., "meta-llama/Llama-2-7b-hf")
llama_model = "meta-llama/Llama-2-7b-hf"
llama_path = os.path.join(base_dir, "llama")
download_and_save(llama_model, llama_path)

# Commented out IPython magic to ensure Python compatibility.
# Step 4: Convert Models to GGUF Format
# %cd /Models/llama.cpp

# For Mistral
!python3 convert_hf_to_gguf.py {mistral_path} --outfile /Models/mistral.gguf --outtype q8_0

# For DeepSeek
!python3 convert_hf_to_gguf.py {deepseek_path} --outfile /Models/deepseek.gguf --outtype q8_0

# For LLaMA
!python3 convert_hf_to_gguf.py {llama_path} --outfile /Models/llama.gguf --outtype q8_0