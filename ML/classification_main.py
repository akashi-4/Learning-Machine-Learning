"""
Main script to run classification experiments.
"""
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.classification.decision_tree import DecisionTreeClassifier
from ML.classification.random_forest import RandomForestClassifier
from ML.classification.knn import KNNClassifier
from ML.classification.svm import SVMClassifier
from ML.classification.neural_network import NeuralNetworkClassifier
from ML.classification.naive_bayes import NaiveBayesClassifier
from ML.classification.logistic_regression import LogisticRegressionClassifier
from ML.utils.metrics import get_best_accuracies
import os
import csv
from datetime import datetime

# Available datasets
DATASETS = {
    'Credit': 'credit.pkl',
    'Census': 'census.pkl'
}

def save_results(name, dataset_name, model_name, accuracy, parameters):
    """Save experiment results to a CSV file.
    
    Args:
        dataset_name (str): Name of the dataset used
        model_name (str): Name of the model/algorithm used
        accuracy (float): Accuracy score achieved
        parameters (dict): Dictionary containing model parameters
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define the results file path
    results_file = os.path.join(results_dir, f'{name}_results.csv')
    
    # Check if file exists to write headers
    file_exists = os.path.isfile(results_file)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format parameters as string
    params_str = '; '.join([f"{k}={v}" for k, v in parameters.items()])
    
    # Write results to CSV
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if file is new
        if not file_exists:
            writer.writerow(['Timestamp', 'Dataset', 'Model', 'Parameters', 'Accuracy'])
        
        # Write results
        writer.writerow([timestamp, dataset_name, model_name, params_str, f"{accuracy:.4f}"])
    
    print(f"\nResults saved to {results_file}")

def choose_dataset():
    """Let user choose a dataset."""
    print('Choose the dataset you want to use:')
    print('1 - Credit')
    print('2 - Census')
    print('3 - Print Best Accuracies (Classification)')
    print('0 - Exit')
    print("-------------------")
    return int(input())

def choose_model():
    """Let user choose a model."""
    print('Choose the model you want to use:')
    print('1 - Logistic Regression')
    print('2 - Naive Bayes')
    print('3 - Decision Tree')
    print('4 - Random Forest')
    print('5 - KNN')
    print('6 - SVM')
    print('7 - Neural Network')
    print('0 - Exit')
    return int(input())

def main():
    """Main function to run experiments."""
    print("-------------------")
    print("Welcome to the Machine Learning Model Selection")
    print("You can choose the dataset and the model you want to use")
    print("The results will be saved in the results folder")
    print("-------------------")

    while True:
        dataset_choice = choose_dataset()
        print("-------------------")

        if dataset_choice == 0:
            break
        elif dataset_choice == 3:
            get_best_accuracies("classification")
            continue

        dataset_name = 'Credit' if dataset_choice == 1 else 'Census'
        filename = DATASETS[dataset_name]
        
        model_choice = choose_model()
        if model_choice == 0:
            break

        # Initialize the chosen model and get parameters
        model = None
        parameters = {}
        
        if model_choice == 1:
            model = LogisticRegressionClassifier()
            parameters = {'solver': 'lbfgs', 'max_iter': 1000}
        elif model_choice == 2:
            model = NaiveBayesClassifier()
            parameters = {'type': 'Gaussian NB'}
        elif model_choice == 3:
            model = DecisionTreeClassifier()
            parameters = {'criterion': 'entropy', 'random_state': 0}
        elif model_choice == 4:
            model = RandomForestClassifier()
            parameters = {'n_estimators': 100, 'criterion': 'entropy', 'random_state': 0}
        elif model_choice == 5:
            model = KNNClassifier()
            parameters = {'n_neighbors': 5, 'metric': 'minkowski', 'p': 2}
        elif model_choice == 6:
            model = SVMClassifier()
            parameters = {'kernel': 'rbf', 'random_state': 0}
        elif model_choice == 7:
            # For Neural Network, get parameters from user
            hidden_layer_sizes, max_iter, tol = NeuralNetworkClassifier.ask_parameters()
            model = NeuralNetworkClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                tol=tol
            )
            parameters = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'max_iter': max_iter,
                'tol': tol,
                'activation': 'relu'
            }

        if model:
            print(f"Training {model.name}...")
            accuracy = model.train_and_evaluate(
                filename,
                show_shapes=True,
                show_confusion=True
            )
            # Save results to CSV
            save_results("classification", dataset_name, model.name, accuracy, parameters)

        print("-------------------")

if __name__ == '__main__':
    main() 