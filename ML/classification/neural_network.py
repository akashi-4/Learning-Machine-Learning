"""
Neural Network Classifier implementation.
"""
from sklearn.neural_network import MLPClassifier as SklearnMLP
from . import BaseClassifier

class NeuralNetworkClassifier(BaseClassifier):
    """
    Neural Network Classifier wrapper class.
    """
    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, tol=0.0001, verbose=False, **kwargs):
        super().__init__("Neural Network")
        self.model = SklearnMLP(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            **kwargs
        )
    
    def fit(self, X, y):
        """Train the neural network model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    @staticmethod
    def ask_parameters():
        """Get neural network parameters from user input."""
        print('Personalize the parameters for the Neural Network')
        print('Hidden Layer Sizes: (Use space to separate the values) use 100 for default')
        if input() == '':
            hidden_layer_sizes = (100,)
        else:
            hidden_layer_sizes = tuple(int(x) for x in input().split())
        print('Max Iter: use 200 for default')
        if input() == '':
            max_iter = 200
        else:
            max_iter = int(input())
        print('Tolerance: use 0.0001 for default')
        if input() == '':
            tol = 0.0001
        else:
            tol = float(input())
        print("-------------------")
        return hidden_layer_sizes, max_iter, tol

# Default hyperparameters for grid search
GRID_PARAMS = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'batch_size': [10, 20, 40],
    'verbose': [False]
}
