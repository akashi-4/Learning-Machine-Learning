"""
Random Forest Regression implementation.
"""
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from . import BaseRegressor

class RandomForestRegressor(BaseRegressor):
    """
    Random Forest Regression wrapper class.
    """
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__("Random Forest Regression")
        self.model = SklearnRF(n_estimators=n_estimators, **kwargs)
    
    def fit(self, X, y):
        """Train the random forest regression model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_

# Default hyperparameters for grid search
GRID_PARAMS = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
} 