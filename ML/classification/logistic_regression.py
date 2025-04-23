"""
Logistic Regression Classifier implementation.
"""
from sklearn.linear_model import LogisticRegression as SklearnLR
from . import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression Classifier wrapper class.
    """
    def __init__(self, **kwargs):
        super().__init__("Logistic Regression")
        self.model = SklearnLR(**kwargs)
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

# Default hyperparameters for grid search
GRID_PARAMS = {
    'C': [1.0, 2.0, 3.0],
    'max_iter': [100, 500, 1000],
    'tol': [0.0001, 0.001, 0.01],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
