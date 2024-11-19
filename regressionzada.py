import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error

def read_csv(filename):
    try:
        path = 'content/{}.csv'.format(filename)
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("File not found")
        return None
    
def preprocess_data(filename):
    path = 'content/{}.csv'.format(filename)
    df = pd.read_csv(path)
    df = correct_null_values(df)

def visualize_corr(df):
    sns.heatmap(df.corr(), annot=True)
    plt.show()

def correct_null_values(df):
    a = df.isnull().sum().sum()
    if a > 0:
        print("Null values found: {}".format(a))
        df.fillna(df.mean(), inplace=True)
    return df
def correlation(X, y):
    corr = np.corrcoef(X, y)
    return corr[0, 1]

# Reshape data to 2D array
def reshape_data(X):
    X = X.reshape(-1, 1)
    return X

def linear_reg(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Get b0 and b1
# b0 = model.intercept_ => where the line crosses the y-axis
# b1 = model.coef_[0] => slope of the line
def get_b0_b1(model):
    b0 = model.intercept_
    b1 = model.coef_[0]
    return b0, b1

def get_predictions(model, X):
    y_pred = model.predict(X)
    return y_pred

def plot_linear_reg(X, y, y_pred):
    grafico = px.scatter(x=X.ravel(), y=y, title='Linear Regression')
    grafico.add_scatter(x=X.ravel(), y=y_pred, mode='lines')
    grafico.show()

def get_score(model, X, y):
    score = model.score(X, y)
    return score

def plot_residuals(model, X, y):
    visualizer = ResidualsPlot(model)
    visualizer.fit(X, y)
    visualizer.poof()

def plano_saude():
    X, y = read_csv('plano_saude')
    corr = correlation(X, y)
    X = reshape_data(X)
    model = linear_reg(X, y)
    b0, b1 = get_b0_b1(model)
    # what is done behind the scenes
    # y_pred = b0 + b1 * X
    y_pred = get_predictions(model, X)
    plot_linear_reg(X, y, y_pred)
    score = get_score(model, X, y)
    plot_residuals(model, X, y)

def house_prices_example_simple_reg(filename):
    house = read_csv(filename=filename)
    X_house = house.iloc[:, 5:6].values
    y_house = house.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(X_house, y_house, test_size=0.3, random_state=0)
    house_model = linear_reg(X_train, y_train)
    y_pred = get_predictions(house_model, X_test)
    plot_linear_reg(X_test, y_test, y_pred)
    score = get_score(house_model, X_test, y_test)
    plot_residuals(house_model, X_test, y_test)
    print(score)
    # Get the difference between the predicted and actual values
    a = abs(y_pred - y_test).mean()
    # or use mean_absolute_error
    b = mean_absolute_error(y_test, y_pred)
    c = mean_squared_error(y_test, y_pred)
    print("Mean absolute error: ", a)
    print("Mean absolute error: ", b)
    print("Mean squared error: ", c)

def house_prices_example_mult_reg(filename):
    house = read_csv(filename=filename)
    X_house = house.iloc[:, 3:19].values
    y_house = house.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(X_house, y_house, test_size=0.3, random_state=0)
    house_model = linear_reg(X_train, y_train)
    y_pred = get_predictions(house_model, X_test)
    score = get_score(house_model, X_test, y_test)
    print(score)
    # Get the difference between the predicted and actual values
    a = mean_absolute_error(y_test, y_pred)
    print("Mean absolute error: ", a)

if __name__ == '__main__':
    #plano_saude()
    #house_prices_example_simple_reg('house_prices')
    house_prices_example_mult_reg('house_prices')