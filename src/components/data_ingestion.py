from src.exception import custom_exception
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
# from src.logger import logging


@dataclass
class DataIngestionConfig:

    data_url = 'https://drive.google.com/file/d/1dL1LV3UYT8mQPX27mO5QYDOSwrrufbdR/view?usp=sharing'
    file_id = data_url.split('/')[-2]
    down_load_url = 'https://drive.google.com/uc?id=' + file_id
    raw_data_path = os.path.join('Artifacts', 'raw.csv')
    train_data_path = os.path.join('Artifacts', 'train.csv')
    test_data_path = os.path.join('Artifacts', 'test.csv')
    

class DataIngestion:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:
            logging.info('Data ingestion has started')

            # Reading dataset from google drive 
            df = pd.read_csv(self.data_ingestion_config.down_load_url)
            logging.info('Dataset Has been read')

            # Saving Raw.csv to Artifacts/raw.csv
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.drop('ID', axis=1, inplace=True)
            df.dropna(inplace=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info('Raw.csv has been save at path : {}'.format(self.data_ingestion_config.raw_data_path))
            logging.info('raw.csv\n{}'.format(df.head()))
            logging.info('Dimesions of the dataset : {}'.format(df.shape))

            # Dividing Datset into Trainning and Testing Data
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=15)
            logging.info('Divide dataset into trainning and testing data')

            # Saving Train and Test data into train.csv and test.csv

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False)

            logging.info('Saved train.csv and test.csv to {} and {} respectively'.format(self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path))


            logging.info('Data Ingestion Complete')
            return (self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path)


        except Exception as e:
            logging.info('Exception Raised at data_ingestion.py')
            raise custom_exception(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(train_path, test_path)