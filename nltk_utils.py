import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
import string

# Download tokenizer if not already present
nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenize a sentence into words
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stem a word to its root form
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag