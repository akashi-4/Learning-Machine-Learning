"""
Support Vector Machine Classifier implementation.
"""
from sklearn.svm import SVC as SklearnSVC
from . import BaseClassifier

class SVMClassifier(BaseClassifier):
    """
    Support Vector Machine Classifier wrapper class.
    """
    def __init__(self, kernel='rbf', C=1.0, random_state=1, **kwargs):
        super().__init__("SVM")
        self.model = SklearnSVC(kernel=kernel, C=C, random_state=random_state, **kwargs)
        self.kernel = kernel
        self.C = C
    
    def fit(self, X, y):
        """Train the SVM model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def train_and_evaluate(self, filename, show_shapes=False, show_confusion=False):
        """
        Override to test different kernels and C values.
        """
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        C_values = [1.0, 2.0]
        
        best_accuracy = 0
        best_config = None
        
        for C in C_values:
            for kernel in kernels:
                self.model = SklearnSVC(kernel=kernel, C=C, random_state=1)
                accuracy = super().train_and_evaluate(filename, show_shapes, show_confusion)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = (kernel, C)
        
        print(f"\nBest configuration: kernel={best_config[0]}, C={best_config[1]}")
        print(f"Best accuracy: {best_accuracy}")
        
        return best_accuracy

# Default hyperparameters for grid search
GRID_PARAMS = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [1.0, 2.0, 3.0]
}
