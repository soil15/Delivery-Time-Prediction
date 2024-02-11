from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainning import ModelTrainning


if __name__ == '__main__':
    
    data_ingestion_obj = DataIngestion()
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_arr, test_arr, _ = data_transformation_obj.initiate_data_transformation(train_path, test_path)

    model_trainning_obj = ModelTrainning()
    model_trainning_obj.initiate_model_trainning(train_arr, test_arr)
    