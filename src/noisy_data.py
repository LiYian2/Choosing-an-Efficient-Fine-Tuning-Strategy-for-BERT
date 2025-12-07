import random
import re
import string
from datasets import Dataset, concatenate_datasets, Features, Value, ClassLabel
from typing import Dict, List

class TextNoiseGenerator:
    """Noisy text generator for data augmentation"""
    
    def __init__(self, seed=42):
        self.emojis = ["😂", "🔥", "😎", "👍", "💯", "👌"]
        self.noise_phrases = ["like, you know", "ummm", "in my opinion, like", 
                              "lol", "I guess", "maybe", "not sure but", "probably"]
        self.sentiment_phrases = ["not good", "very bad", "extremely happy", 
                                  "so sad", "totally awesome", "really terrible"]
        self.methods = [
            self.add_emoji,
            self.introduce_typo_to_sentence,
            self.remove_punctuation,
            self.add_noise_phrase,
            self.add_sentiment_noise
        ]
        random.seed(seed)
    
    def add_emoji(self, text: str) -> str:
        """Add an emoji at a random position in the text"""
        pos = random.randint(0, len(text))
        return text[:pos] + " " + random.choice(self.emojis) + " " + text[pos:]
    
    def introduce_typo(self, word: str) -> str:
        """Introduce a typo in a word"""
        if len(word) < 3:
            return word
        char_idx = random.randint(0, len(word) - 1)
        random_char = random.choice(string.ascii_lowercase)
        return word[:char_idx] + random_char + word[char_idx+1:]
    
    def introduce_typo_to_sentence(self, text: str) -> str:
        """Introduce a typo in a randomly selected word in the sentence"""
        words = text.split()
        if not words:
            return text
        typo_idx = random.randint(0, len(words) - 1)
        words[typo_idx] = self.introduce_typo(words[typo_idx])
        return " ".join(words)
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation and randomly delete words"""
        new_text = re.sub(r'[.,!?;]', '', text)
        words = new_text.split()
        
        if len(words) >= 4:
            remove_count = random.randint(1, 2)
            for _ in range(remove_count):
                if words:  # 防止空列表
                    remove_idx = random.randint(0, len(words) - 1)
                    words.pop(remove_idx)
            new_text = " ".join(words)
        return new_text
    
    def add_noise_phrase(self, text: str) -> str:
        """Add a noise phrase at the beginning"""
        return random.choice(self.noise_phrases) + ", " + text
    
    def add_sentiment_noise(self, text: str) -> str:
        """Add sentiment-related noise at a random position"""
        pos = random.randint(0, len(text))
        return text[:pos] + " " + random.choice(self.sentiment_phrases) + text[pos:]
    
    def generate_noisy_text(self, text: str, min_applications: int = 1) -> str:
        """
        Add noise to the text
        
        Args:
            text: Original text
            min_applications: Minimum number of methods to apply
        
        Returns:
            Noisy text
        """
        noisy_text = text.lower()
        applied_methods = 0
        
        # Randomly apply noise methods
        for method in self.methods:
            if random.random() < 0.35:
                noisy_text = method(noisy_text)
                applied_methods += 1
        
        # Ensure at least a minimum number of methods are applied
        while applied_methods < min_applications:
            method = random.choice(self.methods)
            noisy_text = method(noisy_text)
            applied_methods += 1
        
        return noisy_text

class NoisyDatasetGenerator:
    """Noisy dataset generator"""
    
    def __init__(self, noise_generator=None, seed=42):
        self.noise_generator = noise_generator or TextNoiseGenerator(seed)
        random.seed(seed)
    
    def generate_noisy_sample(self, sentence: str, label: int) -> Dict:
        """Generate a single noisy sample"""
        noisy_sentence = self.noise_generator.generate_noisy_text(sentence)
        return {"sentence": noisy_sentence, "label": label}
    
    def create_noisy_dataset(self, original_dataset, augment_factor=2, 
                           sample_probability=0.1) -> Dataset:
        """
        Create a noisy dataset
        
        Args:
            original_dataset: Original dataset
            augment_factor: Number of noisy samples to generate per original sample
            sample_probability: Probability of selecting a sample to add noise
        
        Returns:
            Noisy dataset
        """
        noisy_sentences = []
        noisy_labels = []
        original_label_names = original_dataset.features["label"].names
        
        for i in range(len(original_dataset)):
            if random.random() < sample_probability:
                sample = original_dataset[i]
                for _ in range(augment_factor):
                    noisy = self.generate_noisy_sample(
                        sample["sentence"], 
                        sample["label"]
                    )
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
    
    def get_robust_train_dataset(self, raw_train_dataset, 
                                augment_factor=2, sample_probability=0.1) -> Dataset:
        """Get robust training dataset (original + noisy)"""
        train_noisy = self.create_noisy_dataset(
            raw_train_dataset, 
            augment_factor=augment_factor,
            sample_probability=sample_probability
        )
        return concatenate_datasets([raw_train_dataset, train_noisy])

# Usage example
def create_robust_dataset(raw_train_dataset, augment_factor=2, seed=42):
    """Convenience function to create a robust dataset"""
    generator = NoisyDatasetGenerator(seed=seed)
    return generator.get_robust_train_dataset(
        raw_train_dataset, 
        augment_factor=augment_factor
    )