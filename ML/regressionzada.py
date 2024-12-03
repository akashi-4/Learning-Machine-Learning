import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

class RegressionAnalyzer:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'polynomial': None,  # Will be created with degree specification
            'decision_tree': DecisionTreeRegressor(),
            'random_forest': RandomForestRegressor(n_estimators=100),
            'svm': SVR(kernel='rbf'),
            'neural_network': MLPRegressor(max_iter=1000, hidden_layer_sizes=(9, 9))
        }
        
    def read_data(self, filename):
        try:
            path = f'content/{filename}.csv'
            return pd.read_csv(path)
        except FileNotFoundError:
            print(f"Error: File {path} not found")
            return None

    def prepare_data(self, df, target_col, feature_cols):
        X = df[feature_cols].values
        y = df[target_col].values
        return train_test_split(X, y, test_size=0.3, random_state=0)

    def scale_data(self, X_train, X_test, y_train, y_test):
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
        
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, sc_y

    def train_and_evaluate(self, model_name, X_train, X_test, y_train, y_test, poly_degree=2, scale_data=False):
        if scale_data:
            X_train, X_test, y_train, y_test, sc_y = self.scale_data(X_train, X_test, y_train, y_test)
        
        if model_name == 'polynomial':
            poly = PolynomialFeatures(degree=poly_degree)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)
            model = LinearRegression()
        else:
            model = self.models[model_name]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if scale_data:
            y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_test = sc_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return metrics, y_pred

    def visualize_results(self, X_test, y_test, y_pred, title):
        if X_test.shape[1] == 1:  # Only plot if we have one feature
            fig = px.scatter(x=X_test.ravel(), y=y_test, title=title)
            fig.add_scatter(x=X_test.ravel(), y=y_pred, mode='lines', name='Regression Line')
            fig.show()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()

def print_menu():
    print("\n=== Regression Analysis Menu ===")
    print("1. Health Insurance Analysis (Plano Saude)")
    print("2. House Price Analysis")
    print("3. Exit")
    
def print_model_menu():
    print("\n=== Model Selection Menu ===")
    print("1. Linear Regression")
    print("2. Polynomial Regression")
    print("3. Decision Tree")
    print("4. Random Forest")
    print("5. SVM")
    print("6. Neural Network")
    print("7. Back to Main Menu")

def main():
    analyzer = RegressionAnalyzer()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-3): ")
        
        if choice == '3':
            print("Goodbye!")
            break
            
        if choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue
            
        # Dataset configuration
        if choice == '1':
            dataset = 'plano_saude'
            target_col = 'cost'
            feature_cols = ['age']
        else:
            dataset = 'house_prices'
            target_col = 'price'
            feature_cols = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 
                          'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                          'sqft_basement', 'yr_built', 'yr_renovated']
        
        # Load data
        df = analyzer.read_data(dataset)
        if df is None:
            continue
            
        # Prepare data
        X_train, X_test, y_train, y_test = analyzer.prepare_data(df, target_col, feature_cols)
        
        while True:
            print_model_menu()
            model_choice = input("Select a model (1-7): ")
            
            if model_choice == '7':
                break
                
            if model_choice not in ['1', '2', '3', '4', '5', '6']:
                print("Invalid choice. Please try again.")
                continue
            
            # Model mapping
            model_mapping = {
                '1': 'linear',
                '2': 'polynomial',
                '3': 'decision_tree',
                '4': 'random_forest',
                '5': 'svm',
                '6': 'neural_network'
            }
            
            model_name = model_mapping[model_choice]
            scale_data = model_name in ['svm', 'neural_network']
            
            # Train and evaluate model
            metrics, y_pred = analyzer.train_and_evaluate(
                model_name, X_train, X_test, y_train, y_test, 
                poly_degree=2, scale_data=scale_data
            )
            
            # Print results
            print("\n=== Model Performance ===")
            print(f"R² Score: {metrics['r2_score']:.4f}")
            print(f"Mean Absolute Error: {metrics['mae']:.4f}")
            print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
            
            # Visualize results
            analyzer.visualize_results(
                X_test, y_test, y_pred,
                f"{model_name.title()} Regression - {dataset}"
            )

if __name__ == "__main__":
    main()