"""
Model evaluation metrics and utilities.
"""
import datetime
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

def compare_predictions(y_test, y_pred):
    """Calculate accuracy score between true and predicted values."""
    return accuracy_score(y_test, y_pred)

def show_confusion_matrix(X_train, Y_train, X_test, Y_test, model):
    """
    Display confusion matrix for the model.
    
    Args:
        X_train: Training features
        Y_train: Training labels
        X_test: Test features
        Y_test: Test labels
        model: Trained classifier model
    """
    cm = ConfusionMatrix(model)
    cm.fit(X_train, Y_train)
    cm.score(X_test, Y_test)
    cm.show()

def save_accuracy(model_name, accuracy, filename):
    """
    Save model accuracy to a CSV file.
    
    Args:
        model_name: Name of the model
        accuracy: Accuracy score
        filename: Name of the dataset file
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('results/accuracy.csv', 'a') as f:
        f.write(f'{date},{model_name},{accuracy},{filename}\n')

def get_best_accuracies():
    """Read and display best accuracies for each model and dataset."""
    with open('results/accuracy.csv', 'r') as f:
        lines = f.readlines()
        best_accuracies = {}
        for line in lines:
            date, model, accuracy, dataset = line.strip().split(',')
            if dataset not in best_accuracies:
                best_accuracies[dataset] = {}
            if model not in best_accuracies[dataset]:
                best_accuracies[dataset][model] = float(accuracy)
            else:
                if float(accuracy) > best_accuracies[dataset][model]:
                    best_accuracies[dataset][model] = float(accuracy)
        return print_best_accuracies(best_accuracies)

def print_best_accuracies(best_accuracies):
    """Print best accuracies in a formatted way."""
    for dataset, models in best_accuracies.items():
        print(f'Best accuracies for dataset {dataset}:')
        for model, accuracy in models.items():
            print(f'{model}: {accuracy:.4f}') 