import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
    




class LLMLayerImportance:
    def __init__(self, model_name="gpt2", device=None):
        """
        Initialize the LLMLayerImportance class to analyze layer importance in LLMs.
        
        Parameters:
        -----------
        model_name : str
            The name of the pretrained model to load (e.g., "gpt2", "llama-7b", "mistral-7b")
        device : torch.device or None
            Device to load the model on. If None, uses CUDA if available, otherwise CPU.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Get model architecture information
        self.num_layers = self.model.config.num_hidden_layers
        self.num_attention_heads = self.model.config.num_attention_heads
        
        # Store reference to whether model uses multi-head attention
        if hasattr(self.model.config, "is_decoder") and self.model.config.is_decoder:
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
            
        # Initialize layer mask attributes
        self.layer_masks = None
        


    def get_dataloader(self, path, name=None, split="validation", batch_size=8, shuffle=False):
        """
        Create a dataloader for evaluating layer importance.
        
        Parameters:
        -----------
        path : str
            Dataset path or name from HuggingFace datasets
        name : str or None
            Specific dataset configuration name
        split : str
            Dataset split to use (e.g., "train", "validation", "test")
        batch_size : int
            Batch size for dataloader
        shuffle : bool
            Whether to shuffle the dataset
            
        Returns:
        --------
        torch.utils.data.DataLoader
            DataLoader containing preprocessed dataset samples
        """

        dataset = load_dataset(path, name, split=split)
        dataset = self._preprocess_dataset(path, dataset)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    


    def _preprocess_dataset(self, path, dataset):
        """
        Preprocess different datasets based on their type.
        
        Parameters:
        -----------
        path : str
            Dataset path or name
        dataset : datasets.Dataset
            Dataset object to preprocess
            
        Returns:
        --------
        datasets.Dataset
            Preprocessed dataset
        """
        if path == "wikitext":
            return self._preprocess_wikitext(dataset)
        elif path == "glue":
            return self._preprocess_glue(dataset)
        elif path == "openai/webtext":
            return self._preprocess_webtext(dataset)
        elif path == "imdb":
            return self._preprocess_imdb(dataset)
        else:
            raise ValueError(f"Preprocessing for dataset {path} is not implemented.")



    def _preprocess_wikitext(self, dataset):
        """Preprocess WikiText dataset for language modeling"""
        def preprocess(batch):
            inputs = self.tokenizer(batch["text"], padding="max_length", 
                                    truncation=True, max_length=512)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
            
        return dataset.map(preprocess, batched=True, remove_columns=["text"])
    


    def _preprocess_glue(self, dataset):
        """Preprocess GLUE dataset for classification"""
        def preprocess(batch):
            if "sentence" in batch:
                texts = batch["sentence"]
            elif "sentence1" in batch and "sentence2" in batch:
                texts = [s1 + " [SEP] " + s2 for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
            else:
                raise ValueError("Unsupported GLUE task format")
                
            inputs = self.tokenizer(texts, padding="max_length", truncation=True, max_length=128)
            inputs["labels"] = batch["label"]
            return inputs
            
        cols_to_remove = [col for col in dataset.column_names if col not in ["label"]]
        return dataset.map(preprocess, batched=True, remove_columns=cols_to_remove)
    


    def _preprocess_webtext(self, dataset):
        """Preprocess WebText dataset for language modeling"""
        def preprocess(batch):
            inputs = self.tokenizer(batch["text"], padding="max_length", 
                                    truncation=True, max_length=512)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
            
        return dataset.map(preprocess, batched=True, remove_columns=["text"])
    


    def _preprocess_imdb(self, dataset):
        """Preprocess IMDB dataset for sentiment classification"""
        def preprocess(batch):
            inputs = self.tokenizer(batch["text"], padding="max_length", 
                                   truncation=True, max_length=512)
            inputs["labels"] = batch["label"]
            return inputs
            
        return dataset.map(preprocess, batched=True, remove_columns=["text"])
    





    def compute_layer_importance(self, dataloader, layer_mask=None, num_batches=50):
        """
        Compute importance scores for layers in the LLM by measuring their influence on model predictions.
        
        Layer importance quantifies how much each layer contributes to the model's performance/loss.
        Higher importance values indicate layers that have greater impact on the model's predictions.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader containing evaluation data
        layer_mask : torch.Tensor or None
            Optional initial mask for model layers of shape (num_layers,)
            Contains 0s and 1s, where 1 indicates the layer is active
        num_batches : int
            Number of batches to use for importance computation
            
        Returns:
        --------
        torch.Tensor
            Normalized importance scores for each layer of shape (num_layers,)
        """

        self.model.eval()
        
        # Initialize layer masks if not provided
        if layer_mask is None:
            layer_mask = torch.ones(self.num_layers, device=self.device, requires_grad=True)
        else:
            # Ensure mask has requires_grad=True for gradient computation
            layer_mask = layer_mask.clone().detach().requires_grad_(True)
            
        # Store the mask for potential later use
        self.layer_masks = layer_mask
        
        # Initialize importance scores
        layer_importance = torch.zeros(self.num_layers, device=self.device)
        
        # Register hooks for each transformer layer to apply masks
        hooks = []
        
        def get_layer_hook(layer_idx):
            def hook(module, input, output):
                # Apply layer mask through scaling
                mask_value = layer_mask[layer_idx]
                # Scale the output by the mask value
                return output * mask_value
            return hook
        
        # Attach hooks to transformer layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h  # GPT-2 style
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            layers = self.model.model.decoder.layers
        elif hasattr(self.model, "decoder"):
            layers = self.model.decoder.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")

        # Update num_layers just in case it was wrong
        self.num_layers = len(layers)

        # Register hooks safely
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(get_layer_hook(i))
            hooks.append(hook)
        
        # Process batches
        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break
                
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
            labels = batch["labels"].to(self.device)
            
            # Reset gradients
            self.model.zero_grad()
            
            # Forward pass with layer masking applied through hooks
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Accumulate gradients (absolute values) for importance scores
            layer_importance += torch.abs(layer_mask.grad.detach())
            
            # Reset the gradient for the next iteration
            layer_mask.grad.zero_()
            
            batch_count += 1
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Normalize importance scores to [0, 1]
        max_importance = layer_importance.max()
        if max_importance > 0:
            layer_importance = layer_importance / max_importance
        
        return layer_importance






    def visualize_layer_importance(self, layer_importance, save_path="layer_importance.png"):
        """
        Visualize layer importance scores as a bar chart.
        
        Parameters:
        -----------
        layer_importance : torch.Tensor
            Tensor of layer importance scores from compute_layer_importance
        save_path : str
            Path to save the visualization
        """

        plt.figure(figsize=(12, 6))
        
        # Convert to numpy for plotting
        importance_np = layer_importance.cpu().numpy()
        
        # Create bar plot with layer indices
        sns.barplot(x=np.arange(self.num_layers), y=importance_np)
        
        plt.title(f"Layer Importance Scores for {self.model.config.model_type}")
        plt.xlabel("Layer Index")
        plt.ylabel("Importance Score")
        plt.xticks(np.arange(0, self.num_layers, max(1, self.num_layers // 10)))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add importance values as text above bars
        for i, val in enumerate(importance_np):
            if i % max(1, self.num_layers // 10) == 0:  # Only show some values to avoid clutter
                plt.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        


        
    def prune_layers(self, layer_importance, pruning_ratio=0.3):
        """
        Create a pruning mask based on layer importance scores.
        
        Parameters:
        -----------
        layer_importance : torch.Tensor
            Tensor of layer importance scores
        pruning_ratio : float
            Ratio of layers to prune (0.0 to 1.0)
            
        Returns:
        --------
        torch.Tensor
            Binary mask indicating which layers to keep (1) and which to prune (0)
        """

        # Determine number of layers to prune
        num_to_prune = int(self.num_layers * pruning_ratio)
        
        # Create full mask (all 1s initially)
        pruning_mask = torch.ones_like(layer_importance)
        
        if num_to_prune > 0:
            # Get indices of least important layers
            _, indices = torch.topk(layer_importance, self.num_layers - num_to_prune, largest=True)
            
            # Create mask (0 for pruned layers, 1 for kept layers)
            pruning_mask = torch.zeros_like(layer_importance)
            pruning_mask[indices] = 1.0
            
        return pruning_mask
        
    def evaluate_with_mask(self, dataloader, layer_mask, num_eval_batches=50):
        """
        Evaluate model performance with a given layer mask.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader containing evaluation data
        layer_mask : torch.Tensor
            Binary mask indicating which layers to use
        num_eval_batches : int
            Number of batches to use for evaluation
            
        Returns:
        --------
        float
            Average loss on the evaluation data
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        # Register hooks for each transformer layer to apply masks
        hooks = []
        
        def get_layer_hook(layer_idx):
            def hook(module, input, output):
                # Apply layer mask - if mask is 0, effectively disables the layer
                mask_value = layer_mask[layer_idx]
                if mask_value == 0:
                    # For pruned layer, pass through the input directly
                    return input[0]  # Most transformers return a tuple where first element is the layer input
                else:
                    return output
            return hook
        
        # Attach hooks to transformer layers
        for i in range(self.num_layers):
            if hasattr(self.model, "transformer"):
                layer = self.model.transformer.h[i]
            elif hasattr(self.model, "model"):
                if hasattr(self.model.model, "layers"):
                    layer = self.model.model.layers[i]
                else:
                    layer = self.model.model.decoder.layers[i]
            elif hasattr(self.model, "decoder"):
                layer = self.model.decoder.layers[i]
            elif hasattr(self.model, "layers"):
                layer = self.model.layers[i]
            else:
                raise ValueError(f"Unsupported model architecture: {type(self.model)}")
                
            hook = layer.register_forward_hook(get_layer_hook(i))
            hooks.append(hook)
        
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= num_eval_batches:
                    break
                    
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                batch_count += 1
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_loss / batch_count if batch_count > 0 else float('inf')
    
    def find_optimal_pruning(self, dataloader, importance_scores, pruning_ratios=None, tolerance=0.1):
        """
        Find the optimal pruning ratio by testing different ratios.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader containing evaluation data
        importance_scores : torch.Tensor
            Layer importance scores
        pruning_ratios : list or None
            List of pruning ratios to try. If None, uses default values
        tolerance : float
            Maximum acceptable performance degradation ratio
            
        Returns:
        --------
        tuple
            (optimal_pruning_ratio, optimal_mask, performance_results)
        """

        if pruning_ratios is None:
            pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        # First evaluate the model with no pruning to get baseline performance
        baseline_mask = torch.ones(self.num_layers, device=self.device)
        baseline_loss = self.evaluate_with_mask(dataloader, baseline_mask)
        
        results = []
        optimal_ratio = 0.0
        optimal_mask = baseline_mask
        
        for ratio in pruning_ratios:
            # Generate mask based on importance scores
            mask = self.prune_layers(importance_scores, ratio)
            
            # Evaluate with this mask
            loss = self.evaluate_with_mask(dataloader, mask)
            
            # Calculate relative performance degradation
            relative_degradation = (loss - baseline_loss) / baseline_loss
            
            results.append({
                'pruning_ratio': ratio,
                'loss': loss,
                'relative_degradation': relative_degradation,
                'mask': mask.clone()
            })
            
            # Update optimal if within tolerance and most aggressive pruning so far
            if relative_degradation <= tolerance and ratio > optimal_ratio:
                optimal_ratio = ratio
                optimal_mask = mask.clone()
        
        return optimal_ratio, optimal_mask, results
    
    def visualize_pruning_results(self, results, save_path="pruning_results.png"):
        """
        Visualize the impact of different pruning ratios on model performance.
        
        Parameters:
        -----------
        results : list
            List of dictionaries with pruning results from find_optimal_pruning
        save_path : str
            Path to save the visualization
        """

        ratios = [r['pruning_ratio'] for r in results]
        losses = [r['loss'] for r in results]
        degradations = [r['relative_degradation'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot loss
        ax1.plot(ratios, losses, 'o-', color='blue')
        ax1.set_ylabel('Loss')
        ax1.set_title('Effect of Layer Pruning on Model Performance')
        ax1.grid(True)
        
        # Plot relative degradation
        ax2.plot(ratios, degradations, 'o-', color='red')
        ax2.axhline(y=0.1, color='green', linestyle='--', label='10% Degradation Threshold')
        ax2.set_xlabel('Pruning Ratio')
        ax2.set_ylabel('Relative Performance Degradation')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def apply_permanent_pruning(self, layer_mask):
        """
        Apply permanent pruning to the model by modifying its architecture.
        
        WARNING: This permanently modifies the model architecture and cannot be undone.
        
        Parameters:
        -----------
        layer_mask : torch.Tensor
            Binary mask indicating which layers to keep (1) and which to prune (0)
            
        Returns:
        --------
        model
            The pruned model
        """
 
        # Create a copy of the model to avoid modifying the original
        pruned_model = copy.deepcopy(self.model)
        
        # Get indices of layers to keep
        keep_indices = torch.nonzero(layer_mask).squeeze().tolist()
        
        # Handle the case where only one layer is kept
        if isinstance(keep_indices, int):
            keep_indices = [keep_indices]
        
        # Modify the layer structure based on model type
        if hasattr(pruned_model, "transformer"):
            # GPT-2 style
            pruned_model.transformer.h = torch.nn.ModuleList(
                [pruned_model.transformer.h[i] for i in keep_indices]
            )
            pruned_model.config.num_hidden_layers = len(keep_indices)
            
        elif hasattr(pruned_model, "model"):
            if hasattr(pruned_model.model, "layers"):
                pruned_model.model.layers = torch.nn.ModuleList(
                    [pruned_model.model.layers[i] for i in keep_indices]
                )
            else:
                pruned_model.model.decoder.layers = torch.nn.ModuleList(
                    [pruned_model.model.decoder.layers[i] for i in keep_indices]
                )
            pruned_model.config.num_hidden_layers = len(keep_indices)
            
        elif hasattr(pruned_model, "decoder"):
            pruned_model.decoder.layers = torch.nn.ModuleList(
                [pruned_model.decoder.layers[i] for i in keep_indices]
            )
            pruned_model.config.num_hidden_layers = len(keep_indices)
            
        elif hasattr(pruned_model, "layers"):
            pruned_model.layers = torch.nn.ModuleList(
                [pruned_model.layers[i] for i in keep_indices]
            )
            pruned_model.config.num_hidden_layers = len(keep_indices)
            
        # Update model attributes
        self.model = pruned_model
        self.num_layers = len(keep_indices)
        
        return pruned_model