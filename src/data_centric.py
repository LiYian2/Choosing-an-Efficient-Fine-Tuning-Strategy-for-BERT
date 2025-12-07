import random, numpy as np, re, torch, string, os, shutil
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader

def get_centric_dataset(
    sorted_losses,        # List[(index, loss, label)], sorted by loss!
    dataset,              # original HuggingFace Dataset → to extract examples
    total_size,
    easiest_range=0.1,
    easiest_proportion=0.3,
    hardest_range=0.1,
    hardest_proportion=0.3,
    seed=42
):
    """
    Return a data-centric dataset based on loss sorting.
    
    :param sorted_losses: sorted list of (index, loss, label) tuples
    :param dataset: original HuggingFace Dataset to extract examples from
    :param total_size: total number of examples to select
    :param easiest_range: proportion of dataset considered as easiest
    :param easiest_proportion: proportion of total_size to sample from easiest region
    :param hardest_range: proportion of dataset considered as hardest
    :param hardest_proportion: proportion of total_size to sample from hardest region
    :param seed: random seed for reproducibility
    """
    random.seed(seed)
    n = len(sorted_losses)

    pos_region = [item for item in sorted_losses if item[2] == 1]
    neg_region = [item for item in sorted_losses if item[2] == 0]

    # Compute region boundaries
    easiest_n_region = int(easiest_range * n//2)
    hardest_n_region = int(hardest_range * n//2)

    # Split into three regions

    pos_easiest_region = pos_region[:easiest_n_region]
    pos_hardest_region = pos_region[-hardest_n_region:]
    pos_middle_region = pos_region[easiest_n_region:-hardest_n_region or None]
    neg_easiest_region = neg_region[:easiest_n_region]
    neg_hardest_region = neg_region[-hardest_n_region:]
    neg_middle_region = neg_region[easiest_n_region:-hardest_n_region or None]

    # Target numbers (round to int, ensure sum ≤ total_size)
    easiest_target = int(total_size * easiest_proportion)
    hardest_target = int(total_size * hardest_proportion)
    middle_target = total_size - easiest_target - hardest_target
    if middle_target < 0:
        raise ValueError("easiest_proportion + hardest_proportion > 1")

    easiest_pos_samples = random.sample(pos_easiest_region, easiest_target//2) if easiest_target > 0 else []
    hardest_pos_samples = random.sample(pos_hardest_region, hardest_target//2) if hardest_target > 0 else []
    middle_pos_samples = random.sample(pos_middle_region, (total_size - easiest_target - hardest_target)//2) if middle_target > 0 else []

    easiest_neg_samples = random.sample(neg_easiest_region, easiest_target//2) if easiest_target > 0 else []
    hardest_neg_samples = random.sample(neg_hardest_region, hardest_target//2) if hardest_target > 0 else []
    middle_neg_samples = random.sample(neg_middle_region, (total_size - easiest_target - hardest_target)//2) if middle_target > 0 else []

    # Combine and extract indices
    selected = easiest_pos_samples + hardest_pos_samples + middle_pos_samples + easiest_neg_samples + hardest_neg_samples + middle_neg_samples
    indices = [item[0] for item in selected]
    random.shuffle(indices)
    selected_examples = dataset.select(indices)
    return selected_examples


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy, "f1": f1}

def compute_sample_losses(model, tokenizer, dataset, batch_size=16):
    model.eval()
    model.to("cuda")
    losses = []
    labels = []  
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_sample_loss = loss_fct(outputs.logits, batch["labels"])
            losses.extend(per_sample_loss.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())  
    return list(zip(range(len(losses)), losses, labels))  # (index, loss, label)