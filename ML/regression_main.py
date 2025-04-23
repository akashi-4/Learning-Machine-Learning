"""
Main script to run regression experiments.
"""
from regression.linear import LinearRegressor
from regression.ridge import RidgeRegressor
from regression.lasso import LassoRegressor
from regression.svr import SVRegressor
from regression.random_forest import RandomForestRegressor

# Available datasets
DATASETS = {
    'Housing': 'housing.pkl',
    'Diabetes': 'diabetes.pkl'
}

def choose_dataset():
    """Let user choose a dataset."""
    print('Choose the dataset you want to use:')
    print('1 - Housing')
    print('2 - Diabetes')
    print('0 - Exit')
    print("-------------------")
    return int(input())

def choose_model():
    """Let user choose a model."""
    print('Choose the model you want to use:')
    print('1 - Linear Regression')
    print('2 - Ridge Regression')
    print('3 - Lasso Regression')
    print('4 - Support Vector Regression')
    print('5 - Random Forest Regression')
    print('0 - Exit')
    return int(input())

def get_best_metrics():
    """Display best metrics for each model and dataset."""
    import pandas as pd
    try:
        df = pd.read_csv('results/regression_metrics.csv', 
                        names=['Date', 'Model', 'Dataset', 'MSE', 'RMSE', 'MAE', 'R2'])
        
        # Group by Model and Dataset and get the best R2 score
        best_models = df.loc[df.groupby(['Model', 'Dataset'])['R2'].idxmax()]
        
        print("\nBest metrics for each model and dataset:")
        for _, row in best_models.iterrows():
            print(f"\nModel: {row['Model']}")
            print(f"Dataset: {row['Dataset']}")
            print(f"MSE: {row['MSE']:.6f}")
            print(f"RMSE: {row['RMSE']:.6f}")
            print(f"MAE: {row['MAE']:.6f}")
            print(f"R²: {row['R2']:.6f}")
            
    except FileNotFoundError:
        print("No metrics file found yet.")
    except Exception as e:
        print(f"Error reading metrics: {e}")

def main():
    """Main function to run experiments."""
    print("-------------------")
    print("Welcome to the Regression Model Selection")
    print("You can choose the dataset and the model you want to use")
    print("The results will be saved in the results folder")
    print("-------------------")

    while True:
        dataset_choice = choose_dataset()
        print("-------------------")

        if dataset_choice == 0:
            break

        filename = DATASETS['Housing' if dataset_choice == 1 else 'Diabetes']
        
        model_choice = choose_model()
        if model_choice == 0:
            break

        # Initialize the chosen model
        model = None
        if model_choice == 1:
            model = LinearRegressor()
        elif model_choice == 2:
            model = RidgeRegressor()
        elif model_choice == 3:
            model = LassoRegressor()
        elif model_choice == 4:
            model = SVRegressor()
        elif model_choice == 5:
            model = RandomForestRegressor()

        if model:
            print(f"Training {model.name}...")
            model.train_and_evaluate(filename, show_shapes=True)
            
            # For models with coefficients, display them
            if hasattr(model, 'get_coefficients'):
                coef = model.get_coefficients()
                print("\nModel Coefficients:")
                print(f"Intercept: {coef['intercept']:.4f}")
                print("Feature coefficients:", coef['coefficients'])

        print("-------------------")
        
        # Ask if user wants to see best metrics
        print("Would you like to see the best metrics? (y/n)")
        if input().lower() == 'y':
            get_best_metrics()

if __name__ == '__main__':
    main() 