"""This file will contain the most common classifiers and tuning parameters as well."""
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from .config import classifier

class GoClassify:
    """"""
    def __init__(self, n_jobs=-1, cv=3, scoring='accuracy'):
        self.classifier_config_dict = classifier.classifier_config_dict
        self.n_jobs = n_jobs
        self.cv = cv
        self.scoring = scoring

    def train(self, x_train, y_train):
        best_parameters = {}
        for estimator in self.classifier_config_dict:
            param_grid = self.classifier_config_dict[estimator]
            mdl = eval(estimator)()
            clf = RandomizedSearchCV(mdl, param_grid, n_jobs=self.n_jobs, cv=self.cv, scoring=self.scoring)
            clf.fit(x_train, y_train)
            best_parameters[clf.best_estimator_] = clf.best_score_
        best = max(best_parameters, key=best_parameters.get)
        return best