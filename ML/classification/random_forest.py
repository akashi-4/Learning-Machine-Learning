"""
Random Forest Classifier implementation.
"""
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from . import BaseClassifier

class RandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier wrapper class.
    """
    def __init__(self, n_estimators=100, criterion='entropy', **kwargs):
        super().__init__("Random Forest")
        self.model = SklearnRF(n_estimators=n_estimators, criterion=criterion, **kwargs)
    
    def fit(self, X, y):
        """Train the random forest model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

# Default hyperparameters for grid search
GRID_PARAMS = {
    'n_estimators': [10, 40, 150],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
