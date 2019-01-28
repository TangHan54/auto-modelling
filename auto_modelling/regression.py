"""This file will contain the most common classifiers and tuning parameters as well."""
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import RandomizedSearchCV

from .config import regressor

class GoRegress:
    """"""
    def __init__(self, n_jobs=-1, cv=3, scoring='neg_mean_squared_error'):
        self.regressor_config_dict = regressor.regressor_config_dict
        self.n_jobs = n_jobs
        self.cv = cv
        self.scoring = scoring

    def train(self, x_train, y_train):
        best_parameters = {}
        for estimator in self.regressor_config_dict:
            param_grid = self.regressor_config_dict[estimator]
            mdl = eval(estimator)()
            clf = RandomizedSearchCV(mdl, param_grid, n_jobs=self.n_jobs, cv=self.cv, scoring=self.scoring)
            clf.fit(x_train, y_train)
            best_parameters[clf.best_estimator_] = clf.best_score_
        best = max(best_parameters, key=best_parameters.get)
        return best