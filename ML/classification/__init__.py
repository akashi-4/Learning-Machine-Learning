"""
Classification models package.
"""
from abc import ABC, abstractmethod
from ..utils.data_loader import load_data, show_shape
from ..utils.metrics import compare_predictions, show_confusion_matrix, save_accuracy

class BaseClassifier(ABC):
    def __init__(self, name):
        self.name = name
        
    def train_and_evaluate(self, filename, show_shapes=False, show_confusion=False):
        """
        Train and evaluate the classifier on given dataset.
        
        Args:
            filename (str): Name of the dataset file
            show_shapes (bool): Whether to display data shapes
            show_confusion (bool): Whether to show confusion matrix
        """
        # Load data
        X_train, Y_train, X_test, Y_test = load_data(filename)
        
        # Show shapes if requested
        if show_shapes:
            show_shape(X_train, Y_train)
            show_shape(X_test, Y_test)
        
        # Train model
        self.fit(X_train, Y_train)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = compare_predictions(Y_test, y_pred)
        print(f'Accuracy using {self.name}: {accuracy}')
        
        # Show confusion matrix if requested
        if show_confusion:
            show_confusion_matrix(X_train, Y_train, X_test, Y_test, self)
        
        # Save accuracy
        save_accuracy(self.name, accuracy, filename)
        
        return accuracy
    
    @abstractmethod
    def fit(self, X, y):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass 