"""
Regression models package.
"""
from abc import ABC, abstractmethod
from ..utils.data_loader import load_data, show_shape
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class BaseRegressor(ABC):
    def __init__(self, name):
        self.name = name
        
    def train_and_evaluate(self, filename, show_shapes=False):
        """
        Train and evaluate the regressor on given dataset.
        
        Args:
            filename (str): Name of the dataset file
            show_shapes (bool): Whether to display data shapes
        """
        # Load data
        X_train, y_train, X_test, y_test = load_data(filename)
        
        # Show shapes if requested
        if show_shapes:
            show_shape(X_train, y_train)
            show_shape(X_test, y_test)
        
        # Train model
        self.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print results
        print(f"\nResults for {self.name}:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save metrics
        self._save_metrics(filename, mse, rmse, mae, r2)
        
        return mse, rmse, mae, r2
    
    def _save_metrics(self, filename, mse, rmse, mae, r2):
        """Save regression metrics to a CSV file."""
        import datetime
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('results/regression_metrics.csv', 'a') as f:
            f.write(f'{date},{self.name},{filename},{mse:.6f},{rmse:.6f},{mae:.6f},{r2:.6f}\n')
    
    @abstractmethod
    def fit(self, X, y):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass 