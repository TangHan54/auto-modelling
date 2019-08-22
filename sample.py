import pandas as pd
from sklearn.model_selection import train_test_split
from auto_modelling.classification import GoClassify
from auto_modelling.regression import GoRegress
from auto_modelling.preprocess import DataManager
from auto_modelling.stack import Stack
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

logger.info('reading data...')
train = pd.read_csv('data/train.csv')

y = train['y']
X = train.drop('y', axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size = 0.1)
logger.info('processing data...')
dm = DataManager()
x_train, x_test = dm.drop_sparse_columns(x_train, x_test)
x_train, x_test = dm.process_data(x_train, x_test)

# logger.info('training model...')
# stack = Stack(mode='regression')
# stack.train(x_train, x_test, y_train, y_test)
# clf = GoClassify()
# best = clf.train(x_train, y_train)
# y_pred = best.predict(x_test)
# print(f'accuracy: {accuracy_score(y_pred,y_test)}')

# reg = GoRegress()        
# best_reg = reg.train(x_train,y_train)
# y_pred = best_reg.predict(x_test)
# print(f'mse: {mean_squared_error(y_pred,y_test)}')