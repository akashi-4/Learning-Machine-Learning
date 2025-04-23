"""
Data loading and manipulation utilities.
"""
import pickle
import numpy as np

def load_data(file_name):
    """
    Load data from a pickle file.
    
    Args:
        file_name (str): Name of the file to load from content directory
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test)
    """
    file_name = 'content/' + file_name
    with open(file_name, 'rb') as f:
        X_train, Y_train, X_test, Y_test = pickle.load(f)
        return X_train, Y_train, X_test, Y_test

def show_shape(x, y):
    """Print the shapes of input arrays."""
    print('X shape:', x.shape)
    print('Y shape:', y.shape)

def concatenate_data(X_train, Y_train, X_test, Y_test):
    """
    Concatenate training and test data.
    
    Args:
        X_train: Training features
        Y_train: Training labels
        X_test: Test features
        Y_test: Test labels
        
    Returns:
        tuple: (X, Y) concatenated data
    """
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    return X, Y 