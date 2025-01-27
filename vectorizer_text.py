import requests
import re

from bs4 import BeautifulSoup


import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


import pickle

# Load RHCP lyrics
with open('rhcp.txt', 'r') as file:
    rhcp = file.read().splitlines()

# Load Madonna lyrics
with open('madonna.txt', 'r') as file:
    madonna = file.read().splitlines()



class LyricsVectorizer:
    def __init__(self, stop_words='english', ngram_range=(1, 1)):
        """
        Initialize the preprocessor with CountVectorizer and TfidfTransformer.
        Args:
            stop_words (str): Stop words to use in CountVectorizer (default: 'english').
            ngram_range (tuple): N-gram range for CountVectorizer (default: (1, 1)).
        """
        self.count_vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
        self.tfidf_transformer = TfidfTransformer()
        
    def preprocess(self, corpus_dict):
        all_texts = []
        all_labels = []
        for label, texts in corpus_dict.items():
            all_texts.extend(texts)
            all_labels.extend([label] * len(texts))
        
        # Step 2: Generate the combined term-document matrix
        term_doc_matrix = self.count_vectorizer.fit_transform(all_texts)
        
        # Step 3: Apply TF-IDF transformation
        tfidf_matrix = self.tfidf_transformer.fit_transform(term_doc_matrix)
        
        # Step 4: Create a DataFrame with the TF-IDF matrix
        tfidf_df = pd.DataFrame(
            tfidf_matrix.todense(),
            columns=self.count_vectorizer.get_feature_names_out(),
            index=all_labels
        )
        return tfidf_df.reset_index()
    
    def transform(self, input_texts):
        """
        Transform new data into the same feature space as the training data.
        Args:
            input_texts (list of str): The input text data to be transformed into TF-IDF features.
        Returns:
            scipy.sparse matrix: The transformed input texts in the same feature space as the training data.
        """
        # First, apply CountVectorizer to the new texts
        count_matrix = self.count_vectorizer.transform(input_texts)
        
        # Then, apply TF-IDF transformation to the count matrix
        tfidf_matrix = self.tfidf_transformer.transform(count_matrix)
        
        
        return tfidf_matrix
    

# Initialize the preprocessor
vectorizer = LyricsVectorizer(stop_words='english', ngram_range=(1, 1))


# Preprocess the data
corpus_dict = {
    'rhcp': rhcp,
    'madonna': madonna
}
tfidf_df = vectorizer.preprocess(corpus_dict)

tfidf_df.to_csv("tfidf_df.csv", index=False)

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
