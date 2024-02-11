from src.logger import logging
from src.exception import custom_exception
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import os
import sys
import pandas as pd
import numpy as np
from src.utils import save_obj, evaluate_model



@dataclass
class ModelTrainningConfig:
    
    model_path = os.path.join('Artifacts', 'model.pkl')


class ModelTrainning:

    def __init__(self):
        self.model_trainning_config_obj = ModelTrainningConfig()


    def initiate_model_trainning(self, train_array, test_array):
        
        try:

            model_performance = {}
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear_Regression' : LinearRegression(),
                'Ridge'  : Ridge(),
                'Lasso'  : Lasso(),
                'ElasticNet' : ElasticNet()
            }

            for model_name, model in zip(models.keys(), models.values()):

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                model_performance[model_name] = evaluate_model(y_pred, y_test)


                # print('{}, {}, performance : {}'.format(model_name, model, evaluate_model(y_pred, y_test)))
                # logging.info('{} : \ny_perd : {}\n y_actual : {}'.format(model_name, y_pred, y_test))
    
            logging.info(model_performance)
            best_model = models[max(zip(model_performance.values(), model_performance.keys()))[1]]
            score = max(zip(model_performance.values(), model_performance.keys()))[0]
            logging.info('Best Model : {} r2 Score : {}'.format(best_model, score))

            save_obj(self.model_trainning_config_obj.model_path, best_model)

        except Exception as e:
            logging.info('Exception Raised at ModelTrainning')
            raise  custom_exception(e, sys)