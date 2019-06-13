import joblib
import pandas as pd
import logging

from auto_modelling.classifion import GoClassify
from auto_modelling.regression import GoRegress
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)

class Stack:
    """Choose the best models and use stacking technique to further improve the performance."""

    def __init__(self, mode='classify', n_models=3):
        """
        mode: str. 'classsify' or 'regression'.
        n_models: int. number of models used to stack.
        """
        self.mode = mode
        self.n_models = n_models

    def train(self, x_train, x_test, y_train, y_test, customized_models=[]):
        if self.mode == 'classify':
            model = GoClassify(n_best=self.n_models)
            stack_model = LogisticRegression(n_jobs=-1)
            metric = accuracy_score
        else:
            model = GoRegress(n_best=self.n_models)
            stack_model = LinearRegression(n_jobs=-1)
            metric = mean_squared_error

        bests = model.train(x_train, y_train) + customized_models
        y_preds = [m.predict(x_test) for m in bests]
        y_preds = pd.DataFrame(y_preds).T
        x_tr_stack, x_te_stack, y_tr_stack, y_te_stack = train_test_split(y_preds, y_test, random_state = 666, test_size = 0.3)
        stack_model.fit(x_tr_stack,y_tr_stack)
        y_pred_stack = stack_model.predict(x_te_stack)
        x_te_stack['y_pred_stack'] = y_pred_stack
        logger.info(f'After Stacking - {str(metric)}: {metric(y_te_stack, y_pred_stack)}')
        return bests, stack_model