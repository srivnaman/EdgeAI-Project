import json
import torch
from pipeline import QuantizedModelPipeline
import config


def evaluate_(exp_name, quant_config):
    dataset = config.DATASET
    
    # Storage
    all_results = {}
    # benchmark_datasets = ['squad', 'repliQa', 'triviaQa']

    full_experiment_name = f'Model : {config.MODEL_NAME} | Dataset : {dataset} | Quantaization : {exp_name}'
    print("\n" + "="*80)
    print(f"Running experiment: {full_experiment_name}")
    print("="*80)
    
    # Initialize pipeline
    pipeline = QuantizedModelPipeline(datset_name=dataset, model_name=config.MODEL_NAME, quantization_config=quant_config)
    pipeline.load_quantized_model()

    # Evaluate bametrics on dataset
    metrics = pipeline.evaluate_datset()

    all_results[full_experiment_name] = {
        "evaluation_metrics": {
            "Dataset":metrics['Dataset'],
            "model_size_mb": metrics["model_size_mb"],
            "Average_generation_time": metrics["average_generation_time"],
            "Average_bleu_score": metrics["average_bleu_score"],
            "Average_perplexity" : metrics['average_perplexity']
        }
    }

    # Cleanup memory
    del pipeline
    torch.cuda.empty_cache()

    # Save final results
    with open(config.RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n Finished all experiments. Results saved to {config.RESULTS_FILE}")