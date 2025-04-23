"""
Naive Bayes Classifier implementation.
"""
from sklearn.naive_bayes import GaussianNB as SklearnNB
from . import BaseClassifier

class NaiveBayesClassifier(BaseClassifier):
    """
    Naive Bayes Classifier wrapper class.
    """
    def __init__(self, **kwargs):
        super().__init__("Naive Bayes")
        self.model = SklearnNB(**kwargs)
    
    def fit(self, X, y):
        """Train the naive bayes model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
