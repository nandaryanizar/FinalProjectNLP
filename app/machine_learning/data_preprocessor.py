import numpy as np
import pandas as pd
import logging
logging.basicConfig(filename="process.log", level=logging.INFO)

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPrerocessor:
    # # Constructor
    # def __init__(self):
    #     pass

    # Split data to train and test
    @staticmethod
    def generate_train_and_test_data(data):
        return model_selection.train_test_split(data[1], data[0])

    # Generate ngrams features using sklearn TfidfVectorizer
    @staticmethod
    def generate_tfidf_ngrams(train_x, test_x, analyzer='word', maximum_features=None):
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=maximum_features)

        tfidf_vect_ngram.fit(train_x)

        train_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
        test_tfidf_ngram = tfidf_vect_ngram.transform(test_x)

        return tfidf_vect_ngram, train_tfidf_ngram, test_tfidf_ngram