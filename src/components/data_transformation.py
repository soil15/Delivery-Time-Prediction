import numpy as np
import pandas as pd
import sys
import os
from src.logger import logging
from src.exception import custom_exception
from dataclasses import dataclass
from src.utils import sep_cat_num_cols, process_date_time_features, save_obj, do_binary_encoding
from src.components.data_ingestion import DataIngestion
from src.components.model_trainning import ModelTrainning
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder

@dataclass
class DataTransformationConfig():

    pre_processor_obj_path = os.path.join('Artifacts', 'pre_processor_obj.pkl')
    out_put_feature = 'Time_taken (min)'
    date_time_features = ['Order_Date', 'Time_Orderd', 'Time_Order_picked']


class DataTransformation():

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_pre_processor_obj(self, df: pd.DataFrame):

        try:

            logging.info('Preparing pre_processor_object')

            num_cols, cat_cols = sep_cat_num_cols(df, self.data_transformation_config.out_put_feature)

            logging.info('Numerical Columns : {}'.format(num_cols))
            logging.info('categorical Columns : {}'.format(cat_cols))

            # Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            # Binary Encoding Pipeline
            binary_encoding_pipeline=Pipeline(
                steps=[
                    ('binary_encoder', BinaryEncoder()),
                ]
            )

            pre_processor_object=ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('binary_encoding_pipeline', binary_encoding_pipeline, cat_cols)
            ])

            return pre_processor_object

            logging.info('pre_processor_object created successfully.')


        except Exception as e:
            logging.info(
                'Exception Raised at data_transformation.py -> get_pre_processor_obj')
            raise custom_exception(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:

            out_put_feature = 'Time_taken (min)'
            columns_to_drop = ['Delivery_person_ID', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            input_train_df = train_df.drop(columns=columns_to_drop, axis=1)
            target_train_df = train_df[out_put_feature]

            input_test_df = test_df.drop(columns=columns_to_drop, axis=1)
            target_test_df = test_df[out_put_feature]

            logging.info('Features after dropping and before date time processing {}'.format(list(input_train_df.columns)))

            input_train_df = process_date_time_features(input_train_df, self.data_transformation_config.date_time_features)
            input_test_df = process_date_time_features(input_test_df, self.data_transformation_config.date_time_features)


#-----------Performing Binary Encoding Manually for all the categorical columns------------------------------------------------------------------------------------# 
            
            # _, cat_cols = sep_cat_num_cols(input_train_df, self.data_transformation_config.out_put_feature)
            # input_train_df, _ = do_binary_encoding(cat_cols, input_train_df)
            # input_test_df, _ = do_binary_encoding(cat_cols, input_test_df)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

            pre_processor_object = self.get_pre_processor_obj(input_train_df.head())
            
            logging.info('length of columns after data time processing : {}'.format(len(list(input_test_df.columns))))
            logging.info('columns before transformming data : {}'.format(list(input_test_df.columns)))

            input_train_arr = pre_processor_object.fit_transform(input_train_df)
            input_test_arr = pre_processor_object.transform(input_test_df)
            
            logging.info('Features after dropping and after date time processing {}'.format(list(input_train_df.columns)))

            logging.info('input train array : {}\n'.format(input_train_arr))
            logging.info('input test array : {}\n'.format(input_test_arr))

            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            # logging.info(pre_processor_object)

            save_obj(self.data_transformation_config.pre_processor_obj_path, pre_processor_object)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.pre_processor_obj_path
            )

        except Exception as e:

            logging.info('Exception occured at data_transformation.py -> initiate_data_transformation')
            raise custom_exception(e, sys)