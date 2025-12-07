import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm


# ==================== Parser Preparation ====================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_task", type=str, default="sst2")
    parser.add_argument("--target_task", type=str, default="mrpc")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 6657, 2025])
    parser.add_argument("--output_dir", type=str, default="./lora_transfer_results_new_r16")
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()



TARGET_MODULES_CONFIGS = {
    "attn_qkv": ["query", "key", "value"],
    "attn_full": ["query", "key", "value", "attention.output.dense"],
    "ffn_only": ["intermediate.dense", "output.dense"],
    "attn_q_only": ["query"],
    "attn_v_only": ["value"],
    "full_model": ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"],
}

# Labels Mapping for GLUE tasks
NUM_LABELS = {
    "sst2": 2,
    "mrpc": 2,
    "cola": 2,
    "mnli": 3,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
}


# ==================== Data Processing ====================
def preprocess_function(examples, tokenizer, task_name, max_length):
    """Preprocess data based on task type
    Args:
        examples: Input examples from dataset
        tokenizer: Tokenizer instance
        task_name: Name of the GLUE task
        max_length: Maximum sequence length
    Returns:
        Tokenized inputs
    """
    # Single sentence tasks
    if task_name in ["sst2", "cola"]:
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    # Sentence pair tasks
    elif task_name == "mrpc":
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    elif task_name == "qnli":
        return tokenizer(
            examples["question"],
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    elif task_name == "qqp":
        return tokenizer(
            examples["question1"],
            examples["question2"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    elif task_name == "rte":
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )


def load_glue_dataset(task_name, tokenizer, max_length, batch_size, num_workers=4):
    """Load and preprocess GLUE dataset
    Args:
        task_name: Name of the GLUE task
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
    Returns:
        train_loader, eval_loader: DataLoaders for training and evaluation
    """
    dataset = load_dataset("glue", task_name)
    
    # Preprocess the dataset
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, task_name, max_length),
        batched=True,
    )
    
    # Rename label to labels (required by Transformers models)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    
    # Remove unnecessary columns
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in encoded_dataset["train"].column_names:
        columns_to_keep.append("token_type_ids")
    
    columns_to_remove = [col for col in encoded_dataset["train"].column_names if col not in columns_to_keep]
    encoded_dataset = encoded_dataset.remove_columns(columns_to_remove)
    
    # Set format
    encoded_dataset.set_format(type="torch")
    
    # Create DataLoader (with performance optimizations)
    train_loader = DataLoader(
        encoded_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    eval_split = "validation_matched" if task_name == "mnli" else "validation"
    eval_loader = DataLoader(
        encoded_dataset[eval_split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, eval_loader


# ====================  Helper Functions ====================
def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_parameters(model):
    """Print statistics of trainable parameters"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_params
    print(
        f"Trainable parameters: {trainable_params:,} || "
        f"Total parameters: {all_params:,} || "
        f"Trainable percentage: {percentage:.2f}%"
    )
    return trainable_params, all_params, percentage


# ==================== Training Functions ====================
def train_model(model, train_loader, eval_loader, args, device):
    """Train LoRA model on the source task
    args:
        model: The model to train
        train_loader: DataLoader for training data
        eval_loader: DataLoader for evaluation data
        args: Training arguments
        device: Device to use for training
    Returns:
        Trained model
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # AMP GradScaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # AMP forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        
        eval_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Eval Acc={eval_acc:.4f}")
        model.train()
    
    return model


# ==================== SVD Analysis ====================
def analyze_lora_svd(model) -> Dict[str, Dict[str, float]]:
    """Analyze LoRA weights using SVD
    args:
        model: The trained LoRA model
    returns:
        svd_results: Dictionary with SVD metrics for each LoRA layer
    """
    svd_results = {}
    
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # Get LoRA weight matrices
            lora_A = module.lora_A["default"].weight.detach().cpu().numpy()
            lora_B = module.lora_B["default"].weight.detach().cpu().numpy()
            
            # Compute ΔW = B × A
            delta_W = np.matmul(lora_B, lora_A)
            
            # Perform SVD
            U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
            
            # Calculate metrics
            singular_values = S
            mean_sv = float(np.mean(singular_values))
            max_sv = float(np.max(singular_values))
            
            # 95% energy effective rank
            total_energy = np.sum(singular_values ** 2)
            cumsum_energy = np.cumsum(singular_values ** 2)
            rank_95 = int(np.searchsorted(cumsum_energy, 0.95 * total_energy) + 1)
            
            # Stable rank: (Σsi)² / Σ(si²)
            stable_rank = float((np.sum(singular_values) ** 2) / np.sum(singular_values ** 2))
            
            svd_results[name] = {
                "mean_singular_value": mean_sv,
                "max_singular_value": max_sv,
                "rank_95": rank_95,
                "stable_rank": stable_rank,
                "singular_values": singular_values.tolist()[:10],  # Save top 10
            }
    
    return svd_results


# ==================== Zero-Shot Transfer Evaluation ====================
def evaluate_zero_shot(
    source_model,
    target_task,
    tokenizer,
    args,
    device,
) -> float:
    """Zero-shot transfer evaluation
    args:
        source_model: The trained LoRA model on source task
        target_task: Target GLUE task name
        tokenizer: Tokenizer instance
        args: Evaluation arguments
        device: Device to use for evaluation
    returns:
        accuracy: Accuracy on the target task
    """
    # Load the base model for the target task
    target_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS[target_task],
    )
    
    # Get LoRA configuration from the source model
    lora_config = source_model.peft_config["default"]
    
    # Apply LoRA to the target model
    target_model = get_peft_model(target_model, lora_config)
    
    # Copy LoRA weights
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    # Only copy LoRA related weights
    lora_keys = [k for k in source_state_dict.keys() if "lora" in k]
    for key in lora_keys:
        if key in target_state_dict:
            target_state_dict[key] = source_state_dict[key]
    
    target_model.load_state_dict(target_state_dict, strict=False)

    
    # Load target task data
    _, target_eval_loader = load_glue_dataset(
        target_task,
        tokenizer,
        args.max_length,
        args.batch_size,
        num_workers=0,  # No need for multiple workers during evaluation
    )

    # Only fine-tune the classifier head
    for name, param in target_model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
        elif "classifier" in name or "score" in name:
            param.requires_grad = True

    target_model.to(device)
    for epoch in range(3):  # Fine-tune for one epoch
        target_model.train()
        total = 0
        for batch in tqdm(target_eval_loader, desc=f"Fine-tuning classifier on {target_task}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            total += batch["labels"].size(0)
            outputs = target_model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
            for name, param in target_model.named_parameters():
                if param.requires_grad:
                    param.data -= args.learning_rate * param.grad.data
            target_model.zero_grad()

    target_model.eval()
    
    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(target_eval_loader, desc=f"Zero-shot eval on {target_task}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = target_model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    accuracy = correct / total
    return accuracy

def evaluate_linear_probing(
    source_model,
    target_task,
    tokenizer,
    args,
    device,
) -> float:
    """Linear Probing Evaluation: Freeze LoRA, only train the classifier head"""
    
    # 1. Prepare the model
    # Load the base model for the target task (classifier head is randomly initialized)
    target_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS[target_task],
    )
    
    # Get and apply LoRA configuration
    lora_config = source_model.peft_config["default"]
    target_model = get_peft_model(target_model, lora_config)
    
    # Copy LoRA weights
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    lora_keys = [k for k in source_state_dict.keys() if "lora" in k]
    for key in lora_keys:
        if key in target_state_dict:
            target_state_dict[key] = source_state_dict[key]
            
    target_model.load_state_dict(target_state_dict, strict=False)
    target_model.to(device)

    # 2. Freeze LoRA layers
    # Freeze Base + LoRA, and only unfreeze the Classifier
    for name, param in target_model.named_parameters():
        # By default, freeze all
        param.requires_grad = False
        
        # Only unfreeze the classifier head (usually called 'classifier' in BERT, sometimes 'score')
        if "classifier" in name or "score" in name:
            param.requires_grad = True
            
    # 3. Load target task data
    target_train_loader, target_eval_loader = load_glue_dataset(
        target_task,
        tokenizer,
        args.max_length,
        args.batch_size, 
        num_workers=0,
    )

    # 4. Define optimizer (only optimize parameters with requires_grad=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, target_model.parameters()), 
        lr=1e-3 # Linear Probing usually can use a slightly larger LR
    )

    # 5. Fine-tune the classifier head (Few-shot / 1 Epoch)
    target_model.train()
    # total_samples = 0
    # max_samples = 1000 
    
    print(f"Starting Linear Probing on {target_task} (only training classifier head)...")
    for epoch in range(10):
        for batch in tqdm(target_train_loader, desc="Linear Probing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = target_model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 6. Final evaluation (using validation set)
    target_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(target_eval_loader, desc=f"Eval on {target_task}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = target_model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    accuracy = correct / total
    return accuracy

# ==================== Main Process ====================
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Store all results
    all_results = {}
    
    # Iterate over different LoRA configurations
    for config_name, target_modules in TARGET_MODULES_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config_name}")
        print(f"Target modules: {target_modules}")
        print(f"{'='*60}\n")
        
        config_results = {
            "target_modules": target_modules,
            "seeds": {},
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
        }
        
        accuracies = []
        
        # Iterate over different random seeds
        for seed in args.seeds:
            print(f"\n--- Seed: {seed} ---")
            set_seed(seed)
            
            # Step 1: Train on source task
            print(f"Step 1: Training on {args.source_task}...")
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=NUM_LABELS[args.source_task],
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="SEQ_CLS",
            )
            
            model = get_peft_model(model, lora_config)
            print_trainable_parameters(model)
            
            # Load data
            train_loader, eval_loader = load_glue_dataset(
                args.source_task,
                tokenizer,
                args.max_length,
                args.batch_size,
                args.num_workers,
            )
            # Train model
            model = train_model(model, train_loader, eval_loader, args, device)
            
            # Step 2: Zero-shot transfer evaluation
            print(f"\nStep 2: Zero-shot transfer to {args.target_task}...")
            transfer_acc = evaluate_linear_probing(
                model,
                args.target_task,
                tokenizer,
                args,
                device,
            )
            print(f"Transfer Learning Accuracy: {transfer_acc:.4f}")
            accuracies.append(transfer_acc)
            
            # Step 3: SVD analysis
            print("\nStep 3: SVD analysis...")
            svd_results = analyze_lora_svd(model)
            
            # Save results for this seed
            config_results["seeds"][seed] = {
                "transfer_accuracy": transfer_acc,
                "svd_analysis": svd_results,
            }
            
            # Save detailed results
            seed_output_path = os.path.join(
                args.output_dir,
                f"{config_name}_seed{seed}_results.json"
            )
            with open(seed_output_path, "w") as f:
                json.dump(config_results["seeds"][seed], f, indent=2)
        
        # Step 4: Calculate mean and standard deviation
        config_results["mean_accuracy"] = float(np.mean(accuracies))
        config_results["std_accuracy"] = float(np.std(accuracies))
        
        print(f"\nSummary for configuration {config_name}:")
        print(f"  Mean accuracy: {config_results['mean_accuracy']:.4f} ± {config_results['std_accuracy']:.4f}")
        
        all_results[config_name] = config_results
    
    # Save overall results
    summary_path = os.path.join(args.output_dir, "transfer_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Print summary table
    print("\nFinal results summary:")
    print(f"{'Configuration':<20} {'Mean Accuracy':<15} {'Std Dev':<10}")
    print("-" * 50)
    for config_name, results in all_results.items():
        print(f"{config_name:<20} {results['mean_accuracy']:.4f}         ±{results['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()