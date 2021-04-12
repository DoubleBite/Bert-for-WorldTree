"""

Reference: 
"""

import re
import nltk
from nltk.corpus import stopwords


def remove_punctuations(sentence: str):
    """
    Remove the puctuations
    """

    sentence = re.sub(r"[?.,;-]", " ", sentence)  # Sub with 1 space
    sentence = re.sub(r"['`]", "", sentence)  # Sub with 0 space

    return sentence


def preprocess(sentence: str, lemmatizer=None):

    # Remove punctuations
    sentence = remove_punctuations(sentence)

    # Remove stop words
    tmp_words = []
    for word in nltk.word_tokenize(sentence):
        if not word.lower() in stopwords.words("english"):
            tmp_words.append(word)

    # Lemmatize
    if lemmatizer is not None:
        tmp_words = [lemmatizer.lemmatize(word) for word in tmp_words]

    return " ".join(tmp_words)


class WorldTreeLemmatizer:

    def __init__(self, vocab_path="lemmatization-en.txt"):

        self.lemmas = {}

        # Load vocab.
        with open(vocab_path, 'r') as f:
            for line in f:
                line = line.rstrip().lower()
                lemma, word = line.split("\t", maxsplit=1)
                self.lemmas[word] = lemma

    def lemmatize(self, word: str):
        if word.lower() in self.lemmas:
            lemma = self.lemmas[word.lower()]
            return lemma
        else:
            return word.lower()
