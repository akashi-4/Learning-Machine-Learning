"""
K-Nearest Neighbors Classifier implementation.
"""
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from . import BaseClassifier

class KNNClassifier(BaseClassifier):
    """
    K-Nearest Neighbors Classifier wrapper class.
    """
    def __init__(self, n_neighbors=5, metric='minkowski', p=2, **kwargs):
        super().__init__("KNN")
        self.model = SklearnKNN(n_neighbors=n_neighbors, metric=metric, p=p, **kwargs)
    
    def fit(self, X, y):
        """Train the KNN model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

# Default hyperparameters for grid search
GRID_PARAMS = {
    'n_neighbors': [5, 10, 15],
    'metric': ['minkowski', 'euclidean', 'manhattan'],
    'p': [1, 2]
}
