# auto-modelling

This repo is a simple version of parameter tuning.

reference: https://github.com/EpistasisLab/tpot/blob

# Quick set-up

- Clone the repo

- Create the virtual environment
```
mkvirtualenv auto-train
workon auto-train
pip install requirements.txt
```
install `xgboost` 
refrence: 
https://xgboost.readthedocs.io/en/latest/build.html#
https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_on_Mac_OSX?lang=en

# Note

- TO DO: Feature selection, pre-processing

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