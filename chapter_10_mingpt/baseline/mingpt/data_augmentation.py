import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def random_remove_stop_word(text, remove_chance=0.03):
    words = word_tokenize(text)
    filtered_words = []
    for word in words:
        if word in stop_words:
            # Decide to remove this stop word with chance
            if random.random() < 1 - remove_chance:
                continue  # Skip this word, effectively removing it
        filtered_words.append(word)  # Keep the word (whether it is a stop word or not)
    return filtered_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def random_deletion(words, p=0.2):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.random()
        if r > p:
            new_words.append(word)
    return new_words
