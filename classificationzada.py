import pandas as pd
import numpy as np
import pickle
import datetime
import sys
import time

# Avaliação do modelo
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Grid Search e Cross Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score

dict_datasets = {'Credit': 'credit.pkl', 'Census': 'census.pkl'}
dict_int_models = {1: 'LogisticRegression', 2: 'GaussianNB', 3: 'DecisionTreeClassifier', 4: 'RandomForestClassifier', 5: 'KNeighborsClassifier', 6: 'SVC', 7: 'MLPClassifier'}
dict_classifiers = {'LogisticRegression': LogisticRegression, 'GaussianNB': GaussianNB, 'DecisionTreeClassifier': DecisionTreeClassifier, 'RandomForestClassifier': RandomForestClassifier, 'KNeighborsClassifier': KNeighborsClassifier, 'SVC': SVC, 'MLPClassifier': MLPClassifier}
# Carregar os dados
def load_data(file_name):
    file_name = 'content/' + file_name
    with open(file_name, 'rb') as f:
        X_train, Y_train, X_test, Y_test = pickle.load(f)
        return X_train, Y_train, X_test, Y_test
# Funcões de apoio
def ask_to_show_shape():
    print('Do you want to see the shape of the dataset? (y/n)')
    answer = input()
    return answer == 'y'

def type_effect(text, delay=0.07):
    # Split the text into lines first
    lines = text.splitlines()
    
    for i, line in enumerate(lines):
        # Type each character in the line
        for char in line:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
            
        # Add newline if it's not the last line
        if i < len(lines) - 1:
            print()  # This handles the newline properly
        
    print()  # Final newline

def get_best_accuracies():
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
    for dataset, models in best_accuracies.items():
        print(f'Best accuracies for dataset {dataset}:')
        for model, accuracy in models.items():
            print(f'{model}: {accuracy:.4f}')

def ask_to_show_confusion_matrix():
    print('Do you want to see the confusion matrix? (y/n)')
    answer = input()
    print("-------------------")
    return answer == 'y'

def ask_parameters():
    print('Personalize the parameters for the Neural Network')
    print('Hidden Layer Sizes: (Use space to separate the values)')
    hidden_layer_sizes = [int(x) for x in input().split()]
    print('Max Iter:')
    max_iter = int(input())
    print('Tolerance:')
    tol = float(input())
    print("-------------------")
    return hidden_layer_sizes, max_iter, tol, False

def show_shape(x,y):
    print('X shape:', x.shape)
    print('Y shape:', y.shape)

def make_predictions(model, x_test):
    return model.predict(x_test)

def compare_predictions(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def concatenate_data(X_train, Y_train, X_test, Y_test):
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    return X, Y

def show_confusion_matrix(X_train, Y_train, X_test, Y_test, svm_model):
    cm = ConfusionMatrix(svm_model)
    cm.fit(X_train, Y_train)
    cm.score(X_test, Y_test)
    cm.show()

def save_accuracy(model_name, accuracy, filename):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('results/accuracy.csv', 'a') as f:
        f.write(f'{date},{model_name},{accuracy},{filename}\n')
#                   Modelos
# Arvore de decisão
def decision_tree(filename):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    if ask_to_show_shape():
        show_shape(X_train, Y_train)
        show_shape(X_test, Y_test)
    dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    type_effect('Accuracy using Decision Tree: ' + str(accuracy))
    if ask_to_show_confusion_matrix():
        show_confusion_matrix(X_train, Y_train, X_test, Y_test, dt)
    save_accuracy('Decision Tree', accuracy, filename)
# Random Forest
def random_forest(filename):
    X_train, Y_train, X_test, Y_test = load_data(filename)

    if ask_to_show_shape():
        show_shape(X_train, Y_train)
        show_shape(X_test, Y_test)
    random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy')
    random_forest.fit(X_train,Y_train)
    y_pred = random_forest.predict(X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    type_effect('Accuracy using Random Forest: ' + str(accuracy))
    if ask_to_show_confusion_matrix():
        show_confusion_matrix(X_train, Y_train, X_test, Y_test, random_forest)
    save_accuracy('Random Forest', accuracy, filename)
# KNN
def knn(filename):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    if ask_to_show_shape():
        show_shape(X_train, Y_train)
        show_shape(X_test, Y_test)
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    type_effect('Accuracy using KNN: ' + str(accuracy))
    if ask_to_show_confusion_matrix():
        show_confusion_matrix(X_train, Y_train, X_test, Y_test, knn)
    save_accuracy('KNN', accuracy, filename)
# SVM
def train_svm(x_train,y_train,kernel_type, c):
    svm_model = SVC(kernel=kernel_type,random_state=1, C=c)
    svm_model.fit(x_train, y_train)
    return svm_model

def svm(filename):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    if ask_to_show_shape():
        show_shape(X_train, Y_train)
        show_shape(X_test, Y_test)
    print("Using C=1.0")
    svm_model = train_svm(X_train, Y_train, 'linear', 1.0)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('SVM-linear-1', accuracy, filename)
    type_effect('Accuracy using Linear: ' + str(accuracy))
    svm_model = train_svm(X_train, Y_train, 'rbf', 1.0)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('SVM-rbf-1', accuracy, filename)
    type_effect('Accuracy using RBF: ' + str(accuracy))

    svm_model = train_svm(X_train, Y_train, 'poly', 1.0)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred) 
    save_accuracy('SVM-poly-1', accuracy, filename)
    type_effect('Accuracy using Poly: ' + str(accuracy))

    svm_model = train_svm(X_train, Y_train, 'sigmoid', 1.0)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('SVM-sigmoid-1', accuracy, filename)
    type_effect('Accuracy using Sigmoid: ' + str(accuracy))
    print("Using C=2")

    svm_model = train_svm(X_train, Y_train, 'linear', 2)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('SVM-linear-2', accuracy, filename)
    type_effect('Accuracy using Linear: ' + str(accuracy))

    svm_model = train_svm(X_train, Y_train, 'rbf', 2)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('SVM-rbf-2', accuracy, filename)
    type_effect('Accuracy using RBF: ' + str(accuracy))
    svm_model = train_svm(X_train, Y_train, 'poly', 2)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('SVM-poly-2', accuracy, filename)
    type_effect('Accuracy using Poly: ' + str(accuracy))

    svm_model = train_svm(X_train, Y_train, 'sigmoid', 2)
    y_pred = make_predictions(svm_model, X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('SVM-sigmoid-2', accuracy, filename)
    type_effect('Accuracy using Sigmoid: ' + str(accuracy))

# Rede Neural
def neural(filename):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    hidden_layer_sizes, max_iter, tol, verb = ask_parameters()
    if ask_to_show_shape():
        show_shape(X_train, Y_train)
        show_shape(X_test, Y_test)
    mlp = MLPClassifier(verbose=verb, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, tol=tol)
    mlp.fit(X_train, Y_train)
    y_pred = mlp.predict(X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('Neural Network', accuracy, filename)
    type_effect('Accuracy using Neural Network: ' + str(accuracy))
    if ask_to_show_confusion_matrix():
        show_confusion_matrix(X_train, Y_train, X_test, Y_test, mlp)
# Naive Bayes
def naive_bayes(filename):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    if ask_to_show_shape():
        show_shape(X_train, Y_train)
        show_shape(X_test, Y_test)
    naive = GaussianNB()
    naive.fit(X_train, Y_train)
    y_pred = naive.predict(X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    save_accuracy('Naive Bayes', accuracy, filename)
    type_effect('Accuracy using Naive Bayes: ' + str(accuracy))
    if ask_to_show_confusion_matrix():
        show_confusion_matrix(X_train, Y_train, X_test, Y_test, naive)
# Logistic Regression
def logistic_regression(filename):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    if ask_to_show_shape():
        show_shape(X_train, Y_train)
        show_shape(X_test, Y_test)
    logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    y_pred = logistic.predict(X_test)
    accuracy = compare_predictions(Y_test, y_pred)
    type_effect('Accuracy using Logistic Regression: ' + str(accuracy))
    if ask_to_show_confusion_matrix():
        show_confusion_matrix(X_train, Y_train, X_test, Y_test, logistic)
    save_accuracy('Logistic Regression', accuracy, filename)

# Agora vamos utilizar o GridSearchCV para encontrar os melhores parâmetros para cada modelo

def grid_search(filename, modelo, parametro):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    X, Y = concatenate_data(X_train, Y_train, X_test, Y_test)
    grid = GridSearchCV(estimator=modelo, param_grid=parametro)
    grid.fit(X, Y)
    print('Best Score:', grid.best_score_)
    print_best_params(grid.best_params_)
    print("Saving best parameters...")
    save_best_params(modelo.__class__.__name__, grid.best_params_, filename)
    print("Best parameters saved.")
# Definindo os parâmetros
parametros_dt = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10]}
paremetros_rf = {'n_estimators': [10, 40, 150], 'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10]}
parametros_knn = {'n_neighbors': [5, 10, 15], 'metric': ['minkowski', 'euclidean', 'manhattan'], 'p': [1, 2]}
parametros_svm = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [1.0, 2.0, 3.0]}
parametros_mlp = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'batch_size': [10, 20, 40], 'verbose': [False]}
parametros_lr = {'C': [1.0, 2.0, 3.0], 'max_iter': [100, 500, 1000], 'tol': [0.0001, 0.001, 0.01], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
# Vamos salvar os melhores parâmetros para cada modelo
def save_best_params(model, params, dataset):
    # Vamos ler se já há os melhores parâmetros para o modelo para aquele dataset, cada dataset terá um arquivo diferente
    try:
        with open('results/best_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
    except:
        best_params = {}
    # Se não houver, vamos criar um dicionário vazio
    if dataset not in best_params:
        best_params[dataset] = {}
    # Vamos salvar os melhores parâmetros para o modelo
    best_params[dataset][model] = params
    # Vamos salvar o arquivo
    with open('results/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
# Vamos carregar os melhores parâmetros
def extract_best_params(dataset, model_name):
    try:
        with open('results/best_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
            return best_params[dataset][model_name]
    except Exception as e:
        print(f"Error loading best parameters: {e}")
        return None

def cross_validation(filename, model_class, ran):
    best_params = extract_best_params(filename, model_class.__name__)
    
    if best_params is None:
        print("No saved parameters found. Using default parameters.")
        model = model_class()
    else:
        print("Using best parameters:", best_params)
        model = model_class(**best_params)
    
    X_train, Y_train, X_test, Y_test = load_data(filename)
    X, Y = concatenate_data(X_train, Y_train, X_test, Y_test)
    all_results = []
    
    for i in range(ran):
        kf = KFold(n_splits=10, shuffle=True, random_state=i)
        results = cross_val_score(model, X=X, y=Y, cv=kf)
        all_results.extend(results)
    
    # Calculate and print statistics
    results_array = np.array(all_results)
    print('\nCross Validation Results:')
    print(f'Mean Accuracy: {results_array.mean():.4f}')
    print(f'Standard Deviation: {results_array.std():.4f}')
    print(f'Min Accuracy: {results_array.min():.4f}')
    print(f'Max Accuracy: {results_array.max():.4f}')
    
    return results_array

def choose_model():
    print('Choose the model you want to use:')
    print('1 - Logistic Regression')
    print('2 - Naive Bayes')
    print('3 - Decision Tree')
    print('4 - Random Forest')
    print('5 - KNN')
    print('6 - SVM')
    print('7 - Neural Network')
    print('Or choose one of the following options:')
    print('8 - Grid Search')
    print('9 - Cross Validation using KFold')
    print('0 - Exit')
    return int(input())
def choose_model2():
    print('Choose the model you want to apply Grid Search:')
    print('1 - Logistic Regression')
    print('2 - Decision Tree')
    print('3 - Random Forest')
    print('4 - KNN')
    print('5 - SVM')
    print('6 - Neural Network')
    return int(input())
def choose_model3():
    print('Choose the model you want to use:')
    print('1 - Logistic Regression')
    print('2 - Naive Bayes')
    print('3 - Decision Tree')
    print('4 - Random Forest')
    print('5 - KNN')
    print('6 - SVM')
    print('7 - Neural Network')
    print('8 - Print Best Params')
    return int(input())
def choose_dataset():
    print('Choose the dataset you want to use:')
    print('1 - Credit')
    print('2 - Census')
    print('3 - Print Best Accuracies')
    print('0 - Exit')
    print("-------------------")
    return int(input())

def print_best_params(params):
    for key, value in params.items():
        print(f'{key}: {value}')

def main():
    print("-------------------")
    print("Welcome to the Machine Learning Model Selection")
    print("You can choose the dataset and the model you want to use")
    print("You can also apply Grid Search to find the best parameters for the model")
    print("Or you can use Cross Validation to evaluate the model")
    print("The results will be saved in the results folder")
    print("-------------------")
    while (True):
        dataset = choose_dataset()
        print("-------------------")
        if dataset == 1:
            filename = dict_datasets['Credit']
        elif dataset == 2:
            filename = dict_datasets['Census']
        elif dataset == 3:
            get_best_accuracies()
        elif dataset == 0:
            exit()
        menu_option = choose_model()
        if menu_option == 1:
            print("Chosen model: Logistic Regression")
            logistic_regression(filename)
        elif menu_option == 2:
            print("Chosen model: Naive Bayes")
            naive_bayes(filename)
        elif menu_option == 3:
            print("Chosen model: Decision Tree")
            decision_tree(filename)
        elif menu_option == 4:
            print("Chosen model: Random Forest")
            random_forest(filename)
        elif menu_option == 5:
            print("Chosen model: KNN")
            knn(filename)
        elif menu_option == 6:
            print("Chosen model: SVM")
            svm(filename)
        elif menu_option == 7:
            print("Chosen model: Neural Network")
            neural(filename)
        elif menu_option == 8:
            print("-------------------")
            menu_option2 = choose_model2()
            print("Wait a moment, this may take a while")
            print("-------------------")
            if menu_option2 == 1:
                print("Chosen model: Logistic Regression")  
                grid_search(filename, LogisticRegression(), parametros_lr)
            elif menu_option2 == 2:
                print("Chosen model: Decision Tree")
                grid_search(filename, DecisionTreeClassifier(), parametros_dt)
            elif menu_option2 == 3:
                print("Chosen model: Random Forest")
                grid_search(filename, RandomForestClassifier(), paremetros_rf)
            elif menu_option2 == 4:
                print("Chosen model: KNN")
                grid_search(filename, KNeighborsClassifier(), parametros_knn)
            elif menu_option2 == 5:
                print("Chosen model: SVM")
                grid_search(filename, SVC(), parametros_svm)
            elif menu_option2 == 6:
                print("Chosen model: Neural Network")
                grid_search(filename, MLPClassifier(), parametros_mlp)
            print("-------------------")
        elif menu_option == 9:
            print("-------------------")
            menu_option3 = choose_model3()
            model_class = dict_classifiers[dict_int_models[menu_option3]]
            print("-------------------")
            print("Wait a moment, this may take a while")
            print("-------------------")
            results = cross_validation(filename=filename, model_class=model_class, ran=30)
        elif menu_option == 0:
            exit()
        print("-------------------")

def save_params_v2(classifier):
    path = 'results/' + classifier.__class__.__name__ + '_done.sav'
    pickle.dump(classifier, open(path, 'wb'))

classifierdt = DecisionTreeClassifier(criterion='entropy', random_state=0, min_samples_leaf=1, min_samples_split=5, splitter='best')
classifierknn = KNeighborsClassifier(metric='minkowski', n_neighbors=15, p=1)
classifiersvm = SVC(C=3.0, kernel='rbf')
classifierlr = LogisticRegression(C=1.0, max_iter=100, solver='newton-cg', tol=0.0001)
classifiermlp = MLPClassifier(activation='relu', batch_size=40, solver='adam', verbose=False)
classifierforest = RandomForestClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=2, n_estimators=150)


save_params_v2(classifierdt)
save_params_v2(classifierknn)
save_params_v2(classifiersvm)
save_params_v2(classifierlr)
save_params_v2(classifiermlp)
save_params_v2(classifierforest)
print("All models saved!")

if __name__ == '__main__':
    main()


# Utilizando diferentes classificadores para o mesmo cliente, se o numero de classificadores que preverem que o cliente irá pagar for maior que o numero de classificadores que preverem que o cliente não irá pagar, então o cliente irá pagar. Caso contrário, o cliente não irá pagar.
# Podemos criar diferentes pesos para cada classificador, por exemplo, se o classificador 1 for um classificador muito bom, então ele terá um peso maior que os outros classificadores.