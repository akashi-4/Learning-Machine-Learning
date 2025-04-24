"""
Model evaluation metrics and utilities.
"""
import datetime
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix
import pandas as pd
import os

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

def save_accuracy(name,model_name, accuracy, filename):
    """
    Save model accuracy to a CSV file.
    
    Args:
        model_name: Name of the model
        accuracy: Accuracy score
        filename: Name of the dataset file
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'results/{name}_results.csv', 'a') as f:
        f.write(f'{date},{model_name},{accuracy},{filename}\n')

def get_best_accuracies(name):
    """Read and display best accuracies for each model and dataset.
    
    Args:
        name (str): Type of results to display ('classification' or 'regression')
    """
    results_file = os.path.join('results', f'{name}_results.csv')
    
    try:
        # Read the CSV file
        df = pd.read_csv(results_file)
        
        # For each Dataset and Model combination, find the row with max accuracy
        best_indices = df.groupby(['Dataset', 'Model'])['Accuracy'].idxmax()
        best_results = df.loc[best_indices].sort_values(['Dataset', 'Accuracy'], ascending=[True, False])
        
        # Display results
        print(f"\nBest {name.title()} Results:")
        print("=" * 80)
        
        current_dataset = None
        for _, row in best_results.iterrows():
            # Print dataset header if we're starting a new dataset
            if current_dataset != row['Dataset']:
                current_dataset = row['Dataset']
                print(f"\nDataset: {current_dataset}")
                print("-" * 80)
                print("Model                  Accuracy    Parameters")
                print("-" * 80)
            
            # Format and print the result row
            model_name = f"{row['Model']:<20}"
            print(f"{model_name} {row['Accuracy']:.4f}     {row['Parameters']}")
            
        print("-" * 80)
        print("\nNote: Showing best accuracy achieved for each model and dataset combination.")
        
    except FileNotFoundError:
        print(f"\nNo results file found at {results_file}")
        print("Please run some experiments first to generate results.")
    except Exception as e:
        print(f"\nError reading results: {str(e)}")
        print("Detailed DataFrame info:")
        if 'df' in locals():
            print("\nColumns in the DataFrame:", df.columns.tolist())
            print("\nFirst few rows of data:")
            print(df.head())