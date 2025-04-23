"""
Ridge Regression implementation.
"""
from sklearn.linear_model import Ridge as SklearnRidge
from . import BaseRegressor

class RidgeRegressor(BaseRegressor):
    """
    Ridge Regression wrapper class.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__("Ridge Regression")
        self.model = SklearnRidge(alpha=alpha, **kwargs)
    
    def fit(self, X, y):
        """Train the ridge regression model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def get_coefficients(self):
        """Get the coefficients of the model."""
        return {
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_
        }

# Default hyperparameters for grid search
GRID_PARAMS = {
    'alpha': [0.1, 1.0, 10.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
} 