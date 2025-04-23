"""
Support Vector Regression implementation.
"""
from sklearn.svm import SVR as SklearnSVR
from . import BaseRegressor

class SVRegressor(BaseRegressor):
    """
    Support Vector Regression wrapper class.
    """
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, **kwargs):
        super().__init__("Support Vector Regression")
        self.model = SklearnSVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
    
    def fit(self, X, y):
        """Train the SVR model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def train_and_evaluate(self, filename, show_shapes=False):
        """
        Override to test different kernels and C values.
        """
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        C_values = [0.1, 1.0, 10.0]
        epsilon_values = [0.1, 0.2, 0.5]
        
        best_mse = float('inf')
        best_config = None
        best_metrics = None
        
        for kernel in kernels:
            for C in C_values:
                for epsilon in epsilon_values:
                    self.model = SklearnSVR(kernel=kernel, C=C, epsilon=epsilon)
                    metrics = super().train_and_evaluate(filename, show_shapes)
                    mse = metrics[0]  # MSE is the first metric
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_config = (kernel, C, epsilon)
                        best_metrics = metrics
        
        print(f"\nBest configuration:")
        print(f"Kernel: {best_config[0]}")
        print(f"C: {best_config[1]}")
        print(f"Epsilon: {best_config[2]}")
        print(f"Best MSE: {best_mse:.6f}")
        
        return best_metrics

# Default hyperparameters for grid search
GRID_PARAMS = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto']
} 