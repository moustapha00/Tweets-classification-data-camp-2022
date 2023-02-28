from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


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
        self.corpus = X['TweetText'].values.copy()
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',
                                  max_features=100, strip_accents='unicode')
        self.vectorizer = vectorizer.fit(self.corpus)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X['TweetText'].values.copy())


def get_estimator():

    feature_extractor = FeatureExtractor()

    classifier = Classifier()

    pipe = make_pipeline(feature_extractor, classifier)
    return pipe
