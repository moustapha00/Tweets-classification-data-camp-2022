from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords



def clean_data(X_df, y_array=None):
    X_df['TweetText'].replace(to_replace = r'[^\w\s]',
                            value = '', regex = True,
                            inplace = True)
    return X_df, y_array

def tokenize_data(X_df, y_array=None):
    texts = X_df['TweetText'].values.copy()
    # Define a list of stop words
    stop_words = set(stopwords.words('english'))
    tokenized_corpus = [nltk.word_tokenize(sentence) for sentence in texts]
    # Corpus_stemmed = [[ps.stem(word) for word in sentence]
    #  for sentence in tokenized_corpus]
    tokenized_corpus = [[word.lower() for word in sentence]
                         for sentence in tokenized_corpus]
    # Remove stopwords
    tokenized_corpus = [[word for word in sentence if word not in stop_words]
                         for sentence in tokenized_corpus]

    X_df['tokenized_Text'] = pd.Series(tokenized_corpus)
    return X_df, y_array

# Load GloVe embeddings
def load_glove_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings

# Transform text into GloVe embeddings
def transform_text_to_glove_embeddings(tokens, embeddings):

    # Compute average GloVe embedding
    embedding_dim = len(embeddings[next(iter(embeddings))])
    embedding_sum = np.zeros(embedding_dim)
    num_embeddings = 0
    for token in tokens:
        if token in embeddings:
            embedding_sum += embeddings[token]
            num_embeddings += 1
    if num_embeddings > 0:
        embedding_avg = embedding_sum / num_embeddings
    else:
        embedding_avg = np.zeros(embedding_dim)

    return embedding_avg

def glove_embedding(X_df, glov_file_path='glove.twitter.27B.25d.txt', y_array=None):
    glove_embeddings = load_glove_embeddings(glov_file_path)
    X_df = clean_data(X_df, y_array=None)
    X_df = tokenize_data(X_df, y_array=None)
    X_df["glove_embeddings"] = X_df["tokenized_Text"].apply(
        lambda x: transform_text_to_glove_embeddings(x, glove_embeddings))
    return X_df, y_array


def tf_idf(X_df, y_array=None):
    corpus = X_df['TweetText'].values.copy()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',
                                  max_features=1000, strip_accents='unicode')
    X_tf = vectorizer.fit_transform(corpus)
    return X_tf, vectorizer


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return glove_embedding(X, glov_file_path='glove.twitter.27B.25d.txt', y_array=None)


def get_estimator():

    feature_extractor = FeatureExtractor()

    classifier = LogisticRegression(max_iter=1000)

    pipe = make_pipeline(feature_extractor, classifier)
    return pipe
