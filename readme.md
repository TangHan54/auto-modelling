# auto-modelling

Auto-modelling is a convenient library to train and tune machine models automatically.

Its main features include the following:

1. preprocessing columns in all datatypes. (numeric, categorical, text)
2. train machine models and tune parameters automatically.
3. return the best model with optimized parameters.

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

reference: https://github.com/EpistasisLab/tpot/blob

# Installation

`pip install auto-modelling`

# Usage Example
```
from auto_modelling.classifion import GoClassify
from auto_modelling.regression import GoRegress
from auto_modelling.preprocess import DataManager

# preprocessing data
dm = DataManager()
train, test = dm.drop_sparse_columns(x_train, x_test)
train, test = dm.process_data(x_train, x_test)

# classification
clf = GoClassify()
best = clf.train(x_train, y_train)
y_pred = best.predict(x_test)

# regression
reg = GoRegress()
best = reg.train(x_train, y_train)
y_pred = best.predict(x_test)
```

There is an example `test.py` in the root directory of this package. run
`python test.py`.

# Development Guide

- Clone the repo

- Create the virtual environment
```
mkvirtualenv auto-train
workon auto-train
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
    - tuning parameters and model selection
    - feature selection
    - return a model with parameters, columns and a script to process x_test 

3. model-evualation
# Other reference

[Packaging your project](https://packaging.python.org/tutorials/packaging-projects/)