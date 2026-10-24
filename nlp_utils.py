import nltk
import numpy as np
import torch
import json
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized, all_words):
    tokenized = [stem(w) for w in tokenized]
    return np.array([1 if w in tokenized else 0 for w in all_words])