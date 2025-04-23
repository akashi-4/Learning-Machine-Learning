"""
Main script to run classification experiments.
"""
from classification.decision_tree import DecisionTreeClassifier
from classification.random_forest import RandomForestClassifier
from classification.knn import KNNClassifier
from classification.svm import SVMClassifier
from classification.neural_network import NeuralNetworkClassifier
from classification.naive_bayes import NaiveBayesClassifier
from classification.logistic_regression import LogisticRegressionClassifier
from utils.metrics import get_best_accuracies

# Available datasets
DATASETS = {
    'Credit': 'credit.pkl',
    'Census': 'census.pkl'
}

def choose_dataset():
    """Let user choose a dataset."""
    print('Choose the dataset you want to use:')
    print('1 - Credit')
    print('2 - Census')
    print('3 - Print Best Accuracies')
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
            get_best_accuracies()
            continue

        filename = DATASETS['Credit' if dataset_choice == 1 else 'Census']
        
        model_choice = choose_model()
        if model_choice == 0:
            break

        # Initialize the chosen model
        model = None
        if model_choice == 1:
            model = LogisticRegressionClassifier()
        elif model_choice == 2:
            model = NaiveBayesClassifier()
        elif model_choice == 3:
            model = DecisionTreeClassifier()
        elif model_choice == 4:
            model = RandomForestClassifier()
        elif model_choice == 5:
            model = KNNClassifier()
        elif model_choice == 6:
            model = SVMClassifier()
        elif model_choice == 7:
            # For Neural Network, get parameters from user
            hidden_layer_sizes, max_iter, tol = NeuralNetworkClassifier.ask_parameters()
            model = NeuralNetworkClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                tol=tol
            )

        if model:
            print(f"Training {model.name}...")
            model.train_and_evaluate(
                filename,
                show_shapes=True,
                show_confusion=True
            )

        print("-------------------")

if __name__ == '__main__':
    main() 