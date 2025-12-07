import os, random, time, evaluate, pandas as pd, torch, shutil, numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, ClassLabel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from sklearn.metrics import f1_score
from config import *
from itertools import product

def set_precision_for_gpu(training_args=None, verbose=False):
    """
    Smart precision selector:
    - T4 / V100 / Turing -> FP16 only (no BF16, Full FP16 is unsafe)
    - A100 / A40 / 3090 / 4060 / 4090 -> BF16
    args: 
        training_args (TrainingArguments, optional): If provided, will set fp16/bf16 flags accordingly.
        verbose (bool): Whether to print out GPU and precision info.
    returns:
        torch_dtype (torch.dtype): Selected torch dtype (torch.float16 or torch.bfloat16).
        precision_str (str): "FP16" or "BF16"
        bf16_supported (bool): Whether BF16 is supported on this GPU.
    """

    assert torch.cuda.is_available(), "CUDA is not available."

    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    
    bf16_supported = major >= 8  # Ampere / Ada+

    if bf16_supported:
        torch_dtype = torch.bfloat16
        use_fp16 = False
        use_bf16 = True
        precision_str = "BF16"
    else:
        torch_dtype = torch.float16
        use_fp16 = True
        use_bf16 = False
        precision_str = "FP16"

    if training_args is not None:
        training_args.fp16 = use_fp16
        training_args.bf16 = use_bf16

    if verbose:
        print("=" * 70)
        print(f"[GPU] {gpu_name}")
        print(f"[Compute Capability] {major}.{minor}")
        print(f"[Selected Precision] {precision_str}")
        print("=" * 70)

    return torch_dtype, precision_str, bf16_supported


# Set global precision
TORCH_DTYPE, PRECISION_STR, BF16_SUPPORTED = set_precision_for_gpu(verbose=True)


def compute_metrics(eval_pred):
    """Calculate accuracy and F1 score using the evaluate library. Only needed for Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy_metric = evaluate.load("glue", "sst2")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    
    f1 = f1_score(labels, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model.
    args: model (torch.nn.Module): The model to inspect.
    returns: float: Percentage of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return 100 * trainable_params / all_param

def build_default_training_args(
    output_dir: str,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int = BATCH_SIZE
):
    """
    Return a default TrainingArguments object with fixed settings.

    args:

        output_dir (str): Output directory for checkpoints and logs.
        seed (int): Random seed.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size per device. Default is BATCH_SIZE in config.py.
    """
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        seed=seed,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=1,
        #load_best_model_at_end=True,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=100,
        warmup_steps=500,
        # Below are default settings from huggingface. Can be overridden if needed.
        report_to="none",
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        fp16=False,
        bf16=False,
    )

    return args

def build_default_lora_config(
    r: int =16,
    lora_alpha: int =32,
    target_modules: list =["query", "key", "value"],
    lora_dropout: float =0.05,
    bias: str ="none",
    task_type: str ="SEQ_CLS",
):
    """
    Return a default LoraConfig object with fixed settings.
    args:
        r (int): LoRA rank.
        lora_alpha (int): LoRA alpha.
        target_modules (list): List of target modules to apply LoRA.
        lora_dropout (float): LoRA dropout rate.
        bias (str): Bias setting for LoRA.
        task_type (str): Task type for LoRA.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )
    return lora_config

def update_training_args(args: TrainingArguments, **updates):
    """
    Update fields in a TrainingArguments object.
    args:
        args (TrainingArguments): The original TrainingArguments object.
        updates: Key-value pairs of fields to update.
    returns:
        TrainingArguments: The updated TrainingArguments object.
    """
    for key, value in updates.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"TrainingArguments has no attribute '{key}'")
    return args

def update_lora_config(lora_config: LoraConfig, **updates):
    """
    Update fields in a LoraConfig object.
    args:
        lora_config (LoraConfig): The original LoraConfig object.
        updates: Key-value pairs of fields to update.
    returns:
        LoraConfig: The updated LoraConfig object.
    """
    for key, value in updates.items():
        if hasattr(lora_config, key):
            setattr(lora_config, key, value)
        else:
            raise ValueError(f"LoraConfig has no attribute '{key}'")
    return lora_config

def _get_model_and_config(
    method: str,
    lr: float,
    seed: int,
    num_labels: int,
    model_name: str,
    epochs: int = EPOCHS,
    training_args: TrainingArguments = None,
    lora_config: LoraConfig = None,
    **kwargs
):
    """
    Initialize model and training arguments based on the specified method.
    args:
        method (str): The training method to use.
        lr (float): Learning rate.
        seed (int): Random seed.
        num_labels (int): Number of labels for classification.
        model_name (str): Pretrained model name or path.
        epochs (int, optional): Number of training epochs. Defaults to EPOCHS.
        training_args (TrainingArguments, optional): Custom training arguments. Defaults to None.
        lora_config (LoraConfig, optional): Custom LoRA configuration. Defaults to None.
        **kwargs: Additional keyword arguments.
    returns:
        Tuple[AutoModelForSequenceClassification, TrainingArguments, LoraConfig]: Initialized model, training arguments, and LoRA configuration.
    """
    # Confirm method validity
    assert method in ["full", "full_16bit", "lora_16bit", "lora_8bit", "qlora_4bit"], f"Unknown method: {method}"
    set_seed(seed)

    # Create default training args
    default_training_args = build_default_training_args(
            output_dir=f"./{method}_results",
            seed=seed,
            lr=lr,
            epochs=epochs,
            batch_size=kwargs.pop('per_device_train_batch_size', BATCH_SIZE)
        )
    if training_args is None:
        training_args = default_training_args
    else: 
        training_args = update_training_args(default_training_args, **vars(training_args))


    torch_dtype, precision_str, _ = set_precision_for_gpu()
    vram_info = precision_str

    # Create default LoRA config if needed
    if "lora" in method and lora_config is None:
        lora_config = build_default_lora_config()

    bnb_config = None
    model_kwargs = {"num_labels": num_labels}

    if method == "full":
        model_kwargs["torch_dtype"] = kwargs.pop('torch_dtype', None) 
        vram_info = "FP32" if model_kwargs["torch_dtype"] is None else precision_str

    elif method == "full_16bit":
        if not BF16_SUPPORTED:
            print("⚠️ [WARNING] T4 detected: Full 16-bit is numerically unstable.")
            print("⚠️ Falling back to Full FP32 for reproducibility with FP16 amp.")
            model_kwargs["torch_dtype"] = torch.float32

            training_args.fp16 = False # Full FP32
            training_args.fp16 = True
            training_args.fp16_opt_level = "01"
            training_args.bf16 = False
            vram_info = "FP32 (Forced Fallback on T4)"
        else:
            model_kwargs["torch_dtype"] = TORCH_DTYPE  
            vram_info = PRECISION_STR

    elif method == "lora_16bit":
        model_kwargs["torch_dtype"] = torch_dtype
        vram_info = precision_str

    elif method == "lora_8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = bnb_config
        vram_info = "8-bit"
    
    elif method == "qlora_4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch_dtype, 
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch_dtype
        training_args.gradient_checkpointing = True
        vram_info = "4-bit (NF4)"

    # ---- 5. Load Model ----
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    
    # Apply k-bit training preparation
    if bnb_config is not None:
        if method == "qlora_4bit":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        else:
            model = prepare_model_for_kbit_training(model)

    # Apply LoRA wrapping
    if "lora" in method and lora_config is not None:
        model = get_peft_model(model, lora_config)
        
    return model, training_args, vram_info

# ==============================================================================
# 2. Core Training Execution Function
# ------------------------------------------------------------------------------

def _run_trainer(name, model, training_args, train_dataset, eval_dataset, tokenizer, config, delete_checkpoints=True, bootstrap=False):
    """
    Trainer wrapped Function.
    args:
        name (str): Name of the experiment/run. Served as output directory prefix.
        model (torch.nn.Module): The model to train/evaluate.
        training_args (TrainingArguments): Training arguments.
        train_dataset (Dataset): Training dataset.
        eval_dataset (Dataset): Evaluation dataset.
        tokenizer (PreTrainedTokenizer): Tokenizer for data collator.
        config (dict): Experiment configuration dictionary containing method, lr, seed, r, freeze_layers, etc.
        delete_checkpoints (bool, optional): Whether to delete checkpoints after training. Defaults to True.
        bootstrap (bool, optional): Whether to perform bootstrap evaluation. Defaults to False.
    """
    # Force VRAM cleanup and reset statistics
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    set_seed(config['seed'])
    training_args.output_dir = f"./{name}_results"
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    train_duration = end_time - start_time
    
    # Evaluation
    eval_metrics = trainer.evaluate()
    
    # Bootstrap Evaluation
    bs_metrics = {}
    if bootstrap:
        bs_metrics = bootstrap_metrics(trainer, eval_dataset, n_bootstrap=1000)
        eval_metrics.update(bs_metrics)

    # Results dictionary
    results_entry = {
        "Method": name,
        "VRAM Mode": config['vram_mode'],
        "Seed": config['seed'],
        "r": config.get('r', 16),
        "Learning Rate": config['lr'],
        "Shot Size": config.get('shot_size', len(train_dataset)),
        "Freeze Layers": config.get('freeze_layers', 0),
        "Trainable %": f"{print_trainable_parameters(model):.4f}%",
        "Peak VRAM": peak_vram,
        "Accuracy": f"{eval_metrics['eval_accuracy']:.4f}",
        "F1 Score": f"{eval_metrics['eval_f1']:.4f}",
        "BS Mean Acc": f"{eval_metrics.get('bs_mean_accuracy', -1):.4f}",
        "BS Acc CI (Low)": f"{eval_metrics.get('bs_acc_ci_low', -1):.4f}",
        "BS Acc CI (High)": f"{eval_metrics.get('bs_acc_ci_high', -1):.4f}",
        "BS Mean F1": f"{eval_metrics.get('bs_mean_f1', -1):.4f}",
        "BS F1 CI (Low)": f"{eval_metrics.get('bs_f1_ci_low', -1):.4f}",
        "BS F1 CI (High)": f"{eval_metrics.get('bs_f1_ci_high', -1):.4f}",
        "Training Time (s)": f"{train_duration:.1f}",
        "Time per Epoch (s)": f"{train_duration/training_args.num_train_epochs:.1f}",
    }
    
    print(results_entry)
    
    # Clean up checkpoints
    if delete_checkpoints and os.path.exists(training_args.output_dir):
        shutil.rmtree(training_args.output_dir, ignore_errors=True)

    return results_entry

# ==============================================================================
# 3. Exposed Experiment Functions
# ------------------------------------------------------------------------------

def train_and_evaluate_unified(
    name,
    config,
    results_list,
    train_dataset,
    eval_dataset,
    bootstrap=False,
    delete_checkpoints=True,
    training_args: TrainingArguments = None,
    lora_config: LoraConfig = None,
    **kwargs
):
    """
    Unified training and evaluation entry function.
    Now supports external passing of training_args and lora_config for various experimental comparisons.
    args:
        name (str): Name of the experiment/run. Served as output directory prefix.
        config (dict): Experiment configuration dictionary containing method, lr, seed, and optional r(doesnot work, only to record data), freeze_layers(default 0), etc.
    """
    # if using other datasets
    set_seed(config['seed'])
    num_labels = kwargs.pop('num_labels', NUM_LABELS)
    # 1. Initialize model and configuration
    model, training_args, vram_mode = _get_model_and_config(
        config['method'],
        config['lr'],
        config['seed'],
        num_labels,
        MODEL_NAME,
        #EPOCHS,
        training_args=training_args,
        lora_config=lora_config,
        **kwargs
    )
    config['vram_mode'] = vram_mode
    
    # 2. Apply layer freezing (if configured)
    if config.get('freeze_layers', 0) > 0:
        model = freeze_model_parameters(model, config['method'], config['freeze_layers'])

    # 3. Training and evaluation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    entry = _run_trainer(
        name=name,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=config,
        delete_checkpoints=delete_checkpoints,
        bootstrap=bootstrap
    )
    print(entry)
    results_list.append(entry)
    return results_list


def evaluate_only(model_name, eval_dataset, seed, results_list):
    """
    Evaluation-only function, used for zero-shot or special cases.
    args:
        model_name (str): Pretrained model name or path.
        eval_dataset (Dataset): Evaluation dataset.
        seed (int): Random seed for reproducibility.
        results_list (list): List to append the evaluation results.
    returns:
        list: Updated results_list with evaluation results appended.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_dataset = eval_dataset.map(
        lambda x: tokenizer(x['sentence'], truncation=True, max_length=MAX_LENGTH), 
        batched=True, remove_columns=["sentence", "idx"]
    ).rename_column("label", "labels")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    model.to("cuda")

    training_args = TrainingArguments(
        output_dir="./eval_only_results",
        per_device_eval_batch_size=BATCH_SIZE,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    eval_metrics = trainer.evaluate()
    
    results_entry = {
        "Method": "Zero-Shot Evaluation",
        "Shot Size": 0,
        "Seed": seed,
        "Learning Rate": "N/A",
        "Accuracy": f"{eval_metrics['eval_accuracy']:.4f}",
        "F1 Score": f"{eval_metrics['eval_f1']:.4f}",
    }
    
    print(results_entry)
    results_list.append(results_entry)
    return results_list

def few_shot_train_and_evaluate_unified(name, config, seed, results_list, raw_dataset, delete_checkpoints=True, **kwargs):
    """
    Unified few-shot training and evaluation entry function. Note that only raw_dataset is passed in,
    and the function will handle few-shot sampling and tokenization internally.
    args:
        name (str): Name of the experiment/run. Served as output directory prefix.
        config (dict): Experiment configuration dictionary containing method, lr, shot_size, seed, and optional r(doesnot work, only to record data), freeze_layers(default 0), etc.
        seed (int): Random seed for reproducibility.
        results_list (list): List to append the evaluation results.
        raw_dataset (DatasetDict): Raw dataset containing 'train' and 'validation' splits.
        delete_checkpoints (bool, optional): Whether to delete checkpoints after training. Defaults to True.
    returns:
        list: Updated results_list with few-shot training results appended.
    """
    # 1. Prepare Few-shot dataset
    set_seed(seed)
    
    if config['shot_size'] >= len(raw_dataset['train']):
        few_shot_dataset = raw_dataset['train']
    elif config['shot_size'] == 0:
        return evaluate_only(MODEL_NAME, raw_dataset['validation'], seed, results_list)
    else:
        few_shot_dataset = sample_balanced(raw_dataset['train'], config['shot_size'], seed=seed)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_few_shot = few_shot_dataset.map(
        lambda x: tokenizer(x['sentence'], truncation=True, max_length=MAX_LENGTH), 
        batched=True, remove_columns=["sentence", "idx"]
    ).rename_column("label", "labels")
    
    eval_dataset = raw_dataset['validation'].map(
        lambda x: tokenizer(x['sentence'], truncation=True, max_length=MAX_LENGTH), 
        batched=True, remove_columns=["sentence", "idx"]
    ).rename_column("label", "labels")
    
    # 2. Initialize model and configuration
    model, training_args, vram_mode = _get_model_and_config(
        config['method'], config['lr'], seed, NUM_LABELS, MODEL_NAME, config.get('epochs', FEW_SHOT_EPOCHS), 
        per_device_train_batch_size=config.get('batch_size', FEW_SHOT_BATCH_SIZE),
        per_device_eval_batch_size=config.get('batch_size', FEW_SHOT_BATCH_SIZE),
        save_strategy="no",
        load_best_model_at_end=False,
    )
    config['vram_mode'] = vram_mode
    config['seed'] = seed 
    
    # 3. Training and evaluation
    entry = _run_trainer(
        name=f"{name}_fewshot_{config['shot_size']}",
        model=model,
        training_args=training_args,
        train_dataset=tokenized_few_shot,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=config,
        delete_checkpoints=delete_checkpoints,
        bootstrap=False
    )
    # Simplify few-shot results recording
    few_shot_entry = {
        "Method": entry['Method'],
        "Shot Size": entry['Shot Size'],
        "Seed": entry['Seed'],
        "Learning Rate": entry['Learning Rate'],
        "Accuracy": entry['Accuracy'],
        "F1 Score": entry['F1 Score'],
        "Batch Size": config.get('batch_size', FEW_SHOT_BATCH_SIZE),
    }
    results_list.append(few_shot_entry)
    return results_list


# ==============================================================================
# 4. Helpers for Data Loading, Sampling, Bootstrap, Freezing, etc.
# ------------------------------------------------------------------------------

def load_and_tokenize_dataset(task=TASK, max_length=MAX_LENGTH, model_name=MODEL_NAME):
    print("\n--- 3. Preparing Data ---")
    if task in ["sst2"]:
        dataset = load_dataset("glue", task)
    else:
        dataset = load_dataset(task)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    return tokenizer, train_dataset, eval_dataset

def sample_balanced(dataset, n, seed):
    set_seed(seed)
    if 'label' in dataset.column_names:
        pos = dataset.filter(lambda x: x['label'] == 1)
        neg = dataset.filter(lambda x: x['label'] == 0)
    elif 'labels' in dataset.column_names:
        pos = dataset.filter(lambda x: x['labels'] == 1)
        neg = dataset.filter(lambda x: x['labels'] == 0)
    else:
        raise ValueError("Dataset must have 'label' or 'labels' column.")
    
    n_pos = min(n // 2, len(pos))
    n_neg = min(n - n_pos, len(neg))
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Not enough samples to create balanced dataset (pos: {len(pos)}, neg: {len(neg)})")
    
    pos_dataset = pos.shuffle(seed=seed).select(range(n_pos))
    neg_dataset = neg.shuffle(seed=seed).select(range(n_neg))
    

    return concatenate_datasets([pos_dataset, neg_dataset]).shuffle(seed=seed)

def bootstrap_metrics(trainer, eval_dataset, n_bootstrap=1000):
    print(f"\n[Bootstrap] Starting {n_bootstrap} resampling iterations...")
    
    predict_output = trainer.predict(eval_dataset)
    labels = np.array(eval_dataset['labels'])
    predictions = np.argmax(predict_output.predictions, axis=1)
    
    n = len(labels)
    accs = []
    f1s = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        sampled_labels = labels[indices]
        sampled_predictions = predictions[indices]
        
        accs.append(np.mean(sampled_labels == sampled_predictions))
        f1s.append(f1_score(sampled_labels, sampled_predictions, average='macro'))
        
    results = {}
    results['bs_mean_accuracy'] = np.mean(accs)
    results['bs_mean_f1'] = np.mean(f1s)
    results['bs_acc_ci_low'] = np.percentile(accs, 2.5)
    results['bs_acc_ci_high'] = np.percentile(accs, 97.5)
    results['bs_f1_ci_low'] = np.percentile(f1s, 2.5)
    results['bs_f1_ci_high'] = np.percentile(f1s, 97.5)
    
    return results

def freeze_model_parameters(model, method: str, n: int):
    if method not in ["full", "lora_16bit"]:
        raise ValueError(f"Unsupported method for freezing: {method}")

    def is_lora_param(name: str, param) -> bool:
        if hasattr(param, "is_lora") and getattr(param, "is_lora"):
            return True
        lname = name.lower()
        return ("lora_" in lname) or (".lora" in lname)

    for name, param in model.named_parameters():
        if is_lora_param(name, param):
            param.requires_grad = True # LoRA parameters are always trainable
            continue

        # Freeze embeddings if n > 0
        if "embeddings." in name and n > 0:
            param.requires_grad = False
            continue

        # Freeze encoder.layer.<idx> where idx < n
        if "encoder.layer." in name:
            try:
                layer_idx_str = name.split("encoder.layer.")[1].split(".")[0]
                layer_idx = int(layer_idx_str)
            except (IndexError, ValueError):
                continue

            if layer_idx < n:
                param.requires_grad = False
            # else: default remains (True for full, False for LoRA backbone)

    return model

def compute_sample_losses(model, tokenizer, dataset, batch_size=16):
    model.eval()
    model.to("cuda")
    losses = []
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_sample_loss = loss_fct(outputs.logits, batch["labels"])
            losses.extend(per_sample_loss.cpu().tolist())
    return list(enumerate(losses))


def run_epoch_control_unified(config, train_dataset, eval_dataset, max_epochs=EPOCHS, delete_checkpoints=True):
    """
    Run training with epoch-by-epoch control, recording metrics after each epoch.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    set_seed(config['seed'])
    
    # 1. Initialize model and configuration
    model, training_args, _ = _get_model_and_config(
        config['method'], config['lr'], config['seed'], NUM_LABELS, MODEL_NAME, epochs=max_epochs
    )
    training_args.output_dir = f"./{config['method']}_overfit_control_results"
    training_args.num_train_epochs = 1 # Train only 1 epoch per loop
    training_args.save_strategy = "no" # Do not save checkpoints
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    results = []
    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch+1}/{max_epochs} ---")
        
        # Train one epoch
        trainer.train()
        
        # Evaluate on validation set
        eval_metrics = trainer.evaluate(metric_key_prefix="eval")
        
        # Evaluate on training set
        train_metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
        
        results.append({
            "Epoch": epoch + 1,
            "train_accuracy": f"{train_metrics['train_accuracy']:.4f}",
            "val_loss": eval_metrics["eval_loss"],
            "val_accuracy": f"{eval_metrics['eval_accuracy']:.4f}",
            "val_f1": f"{eval_metrics['eval_f1']:.4f}",
        })
        print(results[-1])

    # Clean up checkpoints
    if delete_checkpoints and os.path.exists(training_args.output_dir):
        shutil.rmtree(training_args.output_dir, ignore_errors=True)

    return results
