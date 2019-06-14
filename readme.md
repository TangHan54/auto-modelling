# auto-modelling

Auto-modelling is a convenient library to train and tune machine models automatically.

Its main features include the following:

1. preprocessing columns in all datatypes. (numeric, categorical, text)
2. train machine models and tune parameters automatically.
3. return top n best models with optimized parameters.
4. Apply **stacking** technique to combine the n best models returned by the repo or self-determined fitted models together to get an even better result.

The machine learning models include the following:
- Classification:
    - ExtraTreesClassifier
    - RandomForestClassifier
    - KNeighborsClassifier
    - LogisticRegression
    - XGBClassifier
- Regression:
    - ExtraTreesRegressor
    - GradientBoostingRegressor
    - AdaBoostRegressor
    - DecisionTreeRegressor
    - RandomForestRegressor
    - XGBRegressor
- Stack:
    - for classify: LogisticRegression
    - for regression: LinearRegression

reference: https://github.com/EpistasisLab/tpot/blob

# Installation

`pip install auto-modelling`

# Usage Example
```
from auto_modelling.classification import GoClassify
from auto_modelling.regression import GoRegress
from auto_modelling.preprocess import DataManager
from auto_modelling.stack import Stack

# preprocessing data
dm = DataManager()
train, test = dm.drop_sparse_columns(x_train, x_test)
train, test = dm.process_data(x_train, x_test)

# classification
clf = GoClassify(n_best=1)
best = clf.train(x_train, y_train)
y_pred = best.predict(x_test)

# regression
reg = GoRegress(n_best=1)
best = reg.train(x_train, y_train)
y_pred = best.predict(x_test)

# get top 3 best models
clf = GoClassify(n_best=3)
bests = clf.train(x_train, y_train)
y_preds = [m.predict(x_test) for m in bests]

# Stack top 3 best models
stack = Stack(n_models = 3)
level_0_models, level_1_model = stack.train(x_train, y_train, x_test, y_test)
```

There are examples `test.py` and `sample.py` in the root directory of this package. run
`python test.py`/`python sample.py`.

# Development Guide

- Clone the repo

- Create the virtual environment
```
mkvirtualenv auto
workon auto
pip install requirements.txt
```
if you have issues in installing `xgboost` 
refrence: 
https://xgboost.readthedocs.io/en/latest/build.html#
https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_on_Mac_OSX?lang=en

# Note

- TO DO: Feature selection, pre-processing, evaluation metricss

# Thoughts

- Ideally, any dataframe being throw into this repo, it should be processed.

1. pre-processing 

    - drop column that have too many null(Done)
    - fill na for both numeric and non-numeric values(Done)
    - encoded for non-numeric values(Done)
    - scale values if needed
    - balance the dataset if needed

2. model-training

    - mode = `classification`, `regression`, `auto`
    - split data-set
    - tuning parameters and model selection (Done)
    - feature selection
    - return a model with parameters, columns and a script to process x_test
    - stacking with customized fitted models (Done)

3. model-evualation
# Other reference

[Packaging your project](https://packaging.python.org/tutorials/packaging-projects/)