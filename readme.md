# auto-modelling

This repo is a simple version of parameter tuning.

reference: https://github.com/EpistasisLab/tpot/blob

# Installation

`pip install auto-modelling`

# Usage Example
```
from auto_modelling.classifion import GoClassify
from auto_modelling.regression import GoRegress
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

A  `logging.log` file will be created in your project running directory to track the train progress of your model.

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

- Ideally, any dataframe being throw into this repo, it should be 

1. pre-processing 

    - drop column that have too many null
    - fill na for both numeric and non-numeric values
    - encoded for non-numeric values
    - scale values if needed
    - balance the dataset if needed

2. model-training

    - mode = `classification`, `regression`, `auto`
    - split data-set
    - tuning parameters and model selection
    - feature selection
    - return a model with parameters, columns and a script to process x_test 

# Other reference

[Packaging your project](https://packaging.python.org/tutorials/packaging-projects/)