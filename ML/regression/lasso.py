"""
Lasso Regression implementation.
"""
from sklearn.linear_model import Lasso as SklearnLasso
from . import BaseRegressor

class LassoRegressor(BaseRegressor):
    """
    Lasso Regression wrapper class.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__("Lasso Regression")
        self.model = SklearnLasso(alpha=alpha, **kwargs)
    
    def fit(self, X, y):
        """Train the lasso regression model."""
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
    'selection': ['cyclic', 'random'],
    'max_iter': [1000, 2000, 3000],
    'tol': [1e-4, 1e-3, 1e-2]
} 