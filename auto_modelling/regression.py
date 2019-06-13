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
from heapq import nlargest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoRegress:
    
    def __init__(self, n_jobs=-1, cv=3, scoring='neg_mean_squared_error', n_best=1):
        """
        n_best: integer. Return the top n best models.
        """
        assert isinstance(n_best,int), 'n_best must be an integer'
        assert n_best > 0, 'n_best must be a positive integer'

        self.regressor_config_dict = regressor.regressor_config_dict
        self.n_jobs = n_jobs
        self.cv = cv
        self.scoring = scoring
        self.n_best = n_best

    def train(self, x_train, y_train):
        logger.info("Starting to train models")
        best_parameters = {}
        for estimator in self.regressor_config_dict:
            logger.info(f"Starting to train with {estimator}")
            param_grid = self.regressor_config_dict[estimator]
            mdl = eval(estimator)()
            clf = RandomizedSearchCV(mdl, param_grid, n_jobs=self.n_jobs, cv=self.cv, scoring=self.scoring)
            clf.fit(x_train, y_train)
            best_parameters[clf.best_estimator_] = clf.best_score_
            logger.info("==============================================")
            logger.info(f"The current best result is {max(best_parameters.values())}")
            logger.info(f"with {max(best_parameters, key=best_parameters.get)}")
            logger.info("==============================================")
        if self.n_best > 1:
            bests = nlargest(self.n_best, best_parameters, key=best_parameters.get)
            return bests
        else:
            best = max(best_parameters, key=best_parameters.get)
        return best