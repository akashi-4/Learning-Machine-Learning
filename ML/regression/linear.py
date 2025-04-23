"""
Linear Regression implementation.
"""
from sklearn.linear_model import LinearRegression as SklearnLR
from . import BaseRegressor

class LinearRegressor(BaseRegressor):
    """
    Linear Regression wrapper class.
    """
    def __init__(self, **kwargs):
        super().__init__("Linear Regression")
        self.model = SklearnLR(**kwargs)
    
    def fit(self, X, y):
        """Train the linear regression model."""
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