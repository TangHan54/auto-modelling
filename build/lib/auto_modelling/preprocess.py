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
import os
import json
import shutil
import joblib
import pandas as pd
import logging
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class DataManager:

    def __init__(self, directory='data_process_tools'):
        self.directory = directory
        
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
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.mkdir(self.directory)

        logger.info('dealing with numeric columns...')
        columns = {}
        numeric_columns = list(train.select_dtypes(include='number'))
        columns['numeric_columns'] = numeric_columns
        values = train[numeric_columns].median()
        fill_values = {i:j for (i,j) in zip(numeric_columns, list(values))}
        train[numeric_columns] = train[numeric_columns].fillna(values)
        train_nc = csr_matrix(train[numeric_columns].values)
        if isinstance(test, pd.DataFrame):
            test[numeric_columns] = test[numeric_columns].fillna(values)
            test_nc = csr_matrix(test[numeric_columns].values)

        # 2. categorical and bool 
        object_columns = list(train.select_dtypes(include=['bool','object']))
        text_column = []
        for col in object_columns:
            # check if it's a boolean column after drop all null values
            # if train[col].dropna().apply(isinstance,args = [bool]).all():
            if isinstance(train[col].dropna().iloc[0], bool):
                logger.info(f'dealing with bool column {col}...')
                train[col] = train[col].fillna(False)
                fill_values[col] = False
                if isinstance(test, pd.DataFrame):
                    test[col] = test[col].fillna(False)
            else:
                logger.info(f'dealing with object column {col}...')
                value = train[col].mode()[0]
                fill_values[col] = value
                train[col] = train[col].fillna(value)
                if isinstance(test, pd.DataFrame):
                    test[col] = test[col].fillna(value)
            
            train_object_features = csr_matrix(([], ([], [])), shape=(len(train), 0))
            if isinstance(test, pd.DataFrame):
                test_object_features = csr_matrix(([], ([], [])), shape=(len(test), 0))

            # check whether it's a text column
            if (len(train[col].unique()) >= 100) or ((len(train[col].unique()) >= len(train)/3) and (len(train) > 30)):
                logger.info(f'dealing with text column {col}...')
                fill_values[col] = ''
                ttf = TfidfVectorizer(stop_words='english',max_features=100,ngram_range=(1,2))
                train_object_features = hstack([train_object_features, ttf.fit_transform(train[col])])
                if isinstance(test, pd.DataFrame):
                    test_object_features = hstack([test_object_features, ttf.transform(test[col])])
                # dump text encoders
                f = f'{self.directory}/{col}_encoder.joblib'
                joblib.dump(ttf, f)
                text_column.append(col)
                object_columns.remove(col)
        columns['text_columns'] = text_column

        if len(object_columns) > 0:
            encoder = OneHotEncoder()
            logger.info(f'dealing with categorical column {col}...')
            train_object_features = hstack([train_object_features, encoder.fit_transform(train[object_columns])])
            if isinstance(test, pd.DataFrame):
                test_object_features = hstack([test_object_features, encoder.transform(test[object_columns])])
            # dump category encoder
            f = f'{self.directory}/one_hot_encoder.joblib'
            joblib.dump(encoder, f)
            columns['object_columns'] = object_columns

        # dump columns names 
        f = f'{self.directory}/columns.joblib'
        joblib.dump(columns, f)
        # dump fill_na values
        f=  f'{self.directory}/fill_values.joblib'
        joblib.dump(fill_values, f)

        train = hstack([train_nc, train_object_features]).tocsr()
        if isinstance(test, pd.DataFrame):
            test = hstack([test_nc, test_object_features]).tocsr()
            return train, test
        else:
            return train

    def process_predict_data(self, df):
        assert os.path.exists(self.directory), 'Choose the correct directory tools stored.'
        columns_path = f'{self.directory}/columns.joblib'
        assert os.path.exists(columns_path), 'There is no record of required columns.'
        fill_values_path = f'{self.directory}/fill_values.joblib'
        assert os.path.exists(fill_values_path), 'There is no record of fillna values.'

        columns = joblib.load(columns_path)
        fill_values = joblib.load(fill_values_path)

        df = df.fillna(fill_values)
        
        predict_nc = csr_matrix(df[columns['numeric_columns']].values)
        predict_obj_features = csr_matrix(([], ([], [])), shape=(len(df), 0))
        for col in columns['text_columns']:
            f = f'{self.directory}/{col}_encoder.joblib'
            encoder = joblib.load(f)
            predict_obj_features = hstack([predict_obj_features, encoder.transform(df[col])])

        if 'object_columns' in columns.keys():
            f = f'{self.directory}/one_hot_encoder.joblib'
            encoder = joblib.load(f)
            predict_obj_features = hstack([predict_obj_features, encoder.transform(df[columns['object_columns']])])
        
        return hstack([predict_nc, predict_obj_features]).tocsr()


    

