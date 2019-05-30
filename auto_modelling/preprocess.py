"""
This script aims to process data if needed.
1. drop columns if too many null values
2. fill na 
    2.1 fill numeric with median
    2.2 fill boolean with False
    2.3 fill categorical with mode
3. encode for categorical columns
4. vectorize for text columns
"""
import pandas as pd
import logging
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class DataManager:

    def __init__(self):
        pass
    
    def drop_sparse_columns(self, train, test=None):
        """drop columns that have too many null values"""
        
        logger.info('dropping sparse columns...')

        train = train.dropna(axis=1, thresh=int(0.01*len(train)))
        if isinstance(test, pd.DataFrame):
            return train, test[list(train.columns)]
        else:
            return train

    def process_data(self, train, test=None):
        # fill na and encode
        # 1. numeric 
        logger.info('dealing with numeric columns...')
        numeric_columns = list(train.select_dtypes(include='number'))
        train[numeric_columns] = train[numeric_columns].fillna(train[numeric_columns].median())
        train_nc = csr_matrix(train[numeric_columns].values)
        if isinstance(test, pd.DataFrame):
            test[numeric_columns] = test[numeric_columns].fillna(train[numeric_columns].median())
            test_nc = csr_matrix(test[numeric_columns].values)

        # 2. categorical and bool 
        object_columns = list(train.select_dtypes(include=['bool','object']))
        for col in object_columns:
            # check if it's a boolean column after drop all null values
            if train[col].dropna().apply(isinstance,args = [bool]).all():
                logger.info(f'dealing with bool column {col}...')
                train[col] = train[col].fillna(False)
                if isinstance(test, pd.DataFrame):
                    test[col] = test[col].fillna(False)
            else:
                logger.info(f'dealing with object column {col}...')
                train[col] = train[col].fillna(train[col].mode()[0])
                if isinstance(test, pd.DataFrame):
                    test[col] = test[col].fillna(train[col].mode()[0])
            
            train_object_features = csr_matrix(([], ([], [])), shape=(len(train), 0))
            if isinstance(test, pd.DataFrame):
                test_object_features = csr_matrix(([], ([], [])), shape=(len(test), 0))

            # check whether it's a text column
            if (len(train[col].unique()) >= 100) or ((len(train[col].unique()) >= len(train)/3) and (len(train) > 30)):
                logger.info(f'dealing with text column {col}...')
                ttf = TfidfVectorizer(stop_words='english',max_features=1000,ngram_range=(1,2))
                train_object_features = hstack([train_object_features, ttf.fit_transform(train[col])])
                if isinstance(test, pd.DataFrame):
                    test_object_features = hstack([test_object_features, ttf.transform(test[col])])
            else:
                encoder = OneHotEncoder()
                logger.info(f'dealing with categorical column {col}...')
                train_object_features = hstack([train_object_features, encoder.fit_transform(train[[col]])])
                if isinstance(test, pd.DataFrame):
                    test_object_features = hstack([test_object_features, encoder.fit_transform(test[[col]])])

        train = hstack([train_nc, train_object_features]).tocsr()
        if isinstance(test, pd.DataFrame):
            test = hstack([test_nc, test_object_features]).tocsr()
            return train, test
        else:
            return train


        

    

