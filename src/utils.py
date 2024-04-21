import pandas as pd
from src.exception import custom_exception
from src.logger import logging
import sys
import os
import pickle
from sklearn.metrics import r2_score
import category_encoders as ce


def sep_cat_num_cols(df: pd.DataFrame, out_put_feature: str) -> (list(), list()):

    try:

        logging.info('Separating Numerical and Categorical columns')

        num_cols = [col for col in df.columns if df[col].dtype != 'O']
        num_cols = [col for col in num_cols if col != out_put_feature]
        
        cat_cols = [col for col in df.columns if df[col].dtype == 'O']

        logging.info('categorical and numercal columns separated sucsessfully')
        return (num_cols, cat_cols)

    except Exception as e:
        logging.info('Excaption Raised in Utils.py -> sep_cat_num_cols')
        raise custom_exception(e, sys)


def process_date_time_features(df:pd.DataFrame, features)->pd.DataFrame:

    try:

        features_to_drop = []
        new_features_list = []

        for feature in features:      
            
            if '-' in df[feature][0]:
                seperator = '-'
            elif ':' in df[feature][0]:
                seperator = ':'

            size = len(pd.DataFrame(df[feature].str.split(seperator, expand=True)).columns)
            
            new_features = new_list_of_features(size, feature)

            new_features_list += new_features

            df[new_features] = pd.DataFrame(df[feature].str.split(seperator, expand=True))

            features_to_drop.append(feature)


        total_length = len(df)

        for feature in new_features_list:

            null_count = df[feature].isnull().sum()
            if (null_count/total_length >= 0.9):
                features_to_drop.append(feature)

        df.drop(features_to_drop, axis=1, inplace=True)

        return df.copy(deep=True)

    except Exception as e:
        raise custom_exception(e, sys)


def new_list_of_features(size:int, feature:str)->list:

    return [(feature + str(i+1)) for i in range(size)]



if __name__ == '__main__':

    print(new_list_of_features(3, 'feature'))


def save_obj(file_path:str, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise  custom_exception(e, sys)


def evaluate_model(y_pred, y_actual):

    try:

        r2 = r2_score(y_pred, y_actual)

        return (r2)

    except Exception as e:
        logging.info('Exception raised at utils.py->evaluate model')
        raise  custom_exception(e, sys)

def load_obj(file_path):

    try:

        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('Exception occured at utils.py->load_obj')
        raise  custom_exception(e, sys)
    

def do_binary_encoding(features:list, df:pd.DataFrame)->(pd.DataFrame, list):
    
    binary_encoder = ce.BinaryEncoder()

    new_added_features = []

    for feature in features:

        new_feature_list = list(pd.DataFrame(binary_encoder.fit_transform(df[feature])).columns)
        df[new_feature_list] = pd.DataFrame(binary_encoder.fit_transform(df[feature]))

        new_added_features += new_feature_list
                                            
        df.drop(feature, axis=1, inplace=True)

    return (df.copy(deep=True), new_added_features)