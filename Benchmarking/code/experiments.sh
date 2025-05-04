#!/bin/bash


python main.py --model_name 'gpt2' --dataset 'squad'
python main.py --model_name 'gpt2' --dataset 'repliQa'
python main.py --model_name 'gpt2' --dataset 'triviaQa'

python main.py --model_name 'facebook/opt-350m' --dataset 'squad'
python main.py --model_name 'facebook/opt-350m' --dataset 'repliQa'
python main.py --model_name 'facebook/opt-350m' --dataset 'triviaQa'

python main.py --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' --dataset 'squad'
python main.py --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' --dataset 'repliQa'
python main.py --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' --dataset 'triviaQa'

python main.py --model_name 'distilgpt2' --dataset 'squad'
python main.py --model_name 'distilgpt2' --dataset 'repliQa'
python main.py --model_name 'distilgpt2' --dataset 'triviaQa'
