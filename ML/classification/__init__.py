"""
Classification models package.
"""
from ML.utils.data_loader import load_data, show_shape

class BaseClassifier:
    """Base class for all classifiers."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
    
    def train_and_evaluate(self, filename, show_shapes=False, show_confusion=True):
        """Train and evaluate the model.
        
        Args:
            filename (str): Name of the preprocessed data file
            show_shapes (bool): Whether to show data shapes
            show_confusion (bool): Whether to show confusion matrix
            
        Returns:
            float: Accuracy score
        """
        # Load data
        X_train, y_train, X_test, y_test = load_data(filename)
        
        if show_shapes:
            show_shape(X_train, y_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate and return accuracy
        accuracy = (y_pred == y_test).mean()
        print(f"\n{self.name} Accuracy: {accuracy:.4f}")
        
        return accuracy 