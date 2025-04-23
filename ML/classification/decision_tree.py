"""
Decision Tree Classifier implementation.
"""
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from . import BaseClassifier

class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier wrapper class.
    """
    def __init__(self, criterion='entropy', random_state=0, **kwargs):
        super().__init__("Decision Tree")
        self.model = SklearnDT(criterion=criterion, random_state=random_state, **kwargs)
    
    def fit(self, X, y):
        """Train the decision tree model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

# Default hyperparameters for grid search
GRID_PARAMS = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
} 