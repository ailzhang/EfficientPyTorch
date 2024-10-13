import torch
from torch.utils.data import Dataset
from mingpt.utils import CfgNode as CN
from mingpt.data_augmentation import random_remove_stop_word, random_swap, random_deletion
import re
import random

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data, data_aug_chance):
        self.config = config

        self.data_aug_chance = data_aug_chance
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def preprocess_data(self, chunk):
        chunk_size = len(chunk)

        # Normalize text to lower case
        chunk = chunk.lower()

        # Remove HTML tags and non-alphanumeric characters (excluding spaces)
        chunk = re.sub(r'<[^>]+>', '', chunk)
        chunk = re.sub(r'[^a-z0-9\s]', '', chunk)
        # Randomly remove some stopwords (10% chance)
        words = random_remove_stop_word(chunk, remove_chance=self.data_aug_chance)

        if random.random() < self.data_aug_chance:
          # Random swap (swap two pairs of words)
          words = random_swap(words, 2)

        if random.random() < self.data_aug_chance:
          # Random deletion (delete words with a probability of 0.2)
          words = random_deletion(words, 0.2)

        # Continue with further processing or return the result
        processed_chunk = ' '.join(words)

        # Trunc or Pad
        if chunk_size > len(processed_chunk):
          processed_chunk += " " * (chunk_size - len(processed_chunk))
        else:
          processed_chunk = processed_chunk[:chunk_size]
        assert len(processed_chunk) == chunk_size
        return processed_chunk

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        chunk = self.preprocess_data(chunk)

        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]

        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
