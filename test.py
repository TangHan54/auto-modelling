import pandas as pd
from sklearn.model_selection import train_test_split
from auto_modelling.classification import GoClassify
from auto_modelling.regression import GoRegress
from auto_modelling.preprocess import DataManager
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

data = pd.read_csv('data/clean_test_data.csv')

y = data['y']
X = data.drop('y', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size = 0.1)

dm = DataManager()
train, test = dm.drop_sparse_columns(x_train, x_test)
train, test = dm.process_data(x_train, x_test)

# clf = GoClassify()
# best = clf.train(x_train, y_train)
# y_pred = best.predict(x_test)
# print(f'accuracy: {accuracy_score(y_pred,y_test)}')

# reg = GoRegress()        
# best_reg = reg.train(x_train,y_train)
# y_pred = best_reg.predict(x_test)
# print(f'mse: {mean_squared_error(y_pred,y_test)}')