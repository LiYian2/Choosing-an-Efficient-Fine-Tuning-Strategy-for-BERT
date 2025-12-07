from datasets import Dataset, concatenate_datasets, load_dataset, Features, Value, ClassLabel
import pandas as pd
import random
import re
import string
import os 

def add_emoji(text):
    emojis = ["😂", "🔥", "😎", "👍", "💯", "👌"]
    pos = random.randint(0, len(text))
    return text[:pos] + " " + random.choice(emojis) + text[pos:]

def introduce_typo(word):
    if len(word) < 3:
        return word
    char_idx = random.randint(0, len(word) - 1)
    random_char = random.choice(string.ascii_lowercase)
    return word[:char_idx] + random_char + word[char_idx+1:]

def remove_punctuation(text):
    new_text = re.sub(r'[.,!?;]', '', text)
    words = new_text.split()
    if len(words) >= 4:
        remove_count = random.randint(1, 2)
        for _ in range(remove_count):
            remove_idx = random.randint(0, len(words) - 1)
            words.pop(remove_idx)
        new_text = " ".join(words)
    return new_text

def add_noise_phrase(text):
    phrases = ["like, you know", "ummm", "in my opinion, like", "lol", "I guess", "maybe", "not sure but", "probably"]
    return random.choice(phrases) + ", " + text

def add_sentiment_noise(text):
    phrases = ["not good", "very bad", "extremely happy", "so sad", "totally awesome", "really terrible"]
    pos = random.randint(0, len(text))
    return text[:pos] + " " + random.choice(phrases) + text[pos:]
    
methods = ["add_emoji", "introduce_typo", "remove_punctuation", "add_noise_phrase", "add_sentiment_noise"]

def generate_noisy_sample(sentence: str, label: int):
    """
    Generate a noisy version of the input sentence by applying random noise methods.
    Args:
        sentence (str): The original sentence.
        label (int): The label associated with the sentence.
    Returns:
        dict: A dictionary containing the noisy sentence and its label.
    """
    
    
    new_sentence = sentence.lower()
    boolean_flag = False
    # 1. Randomly add emojis
    if random.random() < 0.35:
        new_sentence = add_emoji(new_sentence)
        boolean_flag = True
    # 2. Randomly introduce typos
    if random.random() < 0.35:
        words = new_sentence.split()
        if words:
            typo_idx = random.randint(0, len(words) - 1)
            words[typo_idx] = introduce_typo(words[typo_idx])
            new_sentence = " ".join(words)
            boolean_flag = True
    # 3. Randomly remove punctuation and some words
    if random.random() < 0.35:
        new_sentence = remove_punctuation(new_sentence)
        boolean_flag = True
    # 4. Randomly add colloquial phrases
    if random.random() < 0.35:
        new_sentence = add_noise_phrase(new_sentence)
        boolean_flag = True
    # 5. Randomly add sentiment-related noise phrases
    if random.random() < 0.35:
        new_sentence = add_sentiment_noise(new_sentence)
        boolean_flag = True
    
    if not boolean_flag:
        # Ensure at least one noise method is applied
        method = random.choice(methods)
        new_sentence = globals()[method](new_sentence)
        boolean_flag = True
        
    return {"sentence": new_sentence, "label": label}


def create_noisy_dataset(original_dataset, augment_factor=2, seed=42):
    """
    Create a noisy dataset by augmenting the original dataset with noisy samples.
    
    :param original_dataset: The original dataset to augment.
    :param augment_factor: The number of noisy samples to generate per original sample.
    :param seed: The random seed for reproducibility.
    """
    random.seed(seed)
    noisy_sentences = []
    noisy_labels = []
    original_label_names = original_dataset.features["label"].names

    for i in range(len(original_dataset)):
        sample = original_dataset[i]
        if random.random() < 0.1:
            for _ in range(augment_factor):
                noisy = generate_noisy_sample(sample["sentence"], sample["label"])
                noisy_sentences.append(noisy["sentence"])
                noisy_labels.append(noisy["label"])

    features = Features({
        "sentence": Value("string"),
        "label": ClassLabel(names=original_label_names)
    })
    return Dataset.from_dict({
        "sentence": noisy_sentences,
        "label": noisy_labels
    }, features=features)

def get_robust_train_dataset(raw_train_dataset, augment_factor=2, seed=42):
    random.seed(seed)
    train_clean = raw_train_dataset
    train_noisy = create_noisy_dataset(raw_train_dataset, augment_factor=augment_factor, seed=seed)
    train_robust = concatenate_datasets([train_clean, train_noisy])
    return train_robust

if __name__ == "__main__":
    raw_train = load_dataset("glue", "sst2", split="train")
    robust_train = get_robust_train_dataset(raw_train, augment_factor=2, seed=42)
    print("Original size:", len(raw_train))
    print("Noisy added:", len(robust_train) - len(raw_train))
    print("Robust size:", len(robust_train))
    print("Sample:", robust_train[0])



