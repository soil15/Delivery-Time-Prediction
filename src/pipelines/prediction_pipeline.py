from src.utils import load_obj, process_date_time_features
import pandas as pd
from src.logger import logging
import os
import sys
from src.components.data_transformation import DataTransformationConfig


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        
        data_transformation_obj = DataTransformationConfig()
        
        model_path = os.path.join('Artifacts', 'model.pkl')
        preprocessor_path = os.path.join('Artifacts', 'pre_processor_obj.pkl')
        
        model = load_obj(model_path)
        preprocessor = load_obj(preprocessor_path)
        
        input_df = process_date_time_features(df=data, features=data_transformation_obj.date_time_features)
        
        array = preprocessor.transform(input_df)
        
        pred = model.predict(array)
        
        return pred


class CustomData:
    def __init__(
        self,
        Delivery_person_Age,
        Delivery_person_Ratings,
        Order_Date,
        Time_Orderd,
        Time_Order_picked,
        Weather_conditions,
        Road_traffic_density,
        Vehicle_condition,
        Type_of_order,
        Type_of_vehicle,
        multiple_deliveries,
        Festival,
        City,
    ):
        
        self.delivery_person_age = Delivery_person_Age
        self.delivery_person_rating = Delivery_person_Ratings
        self.order_Date = Order_Date
        self.time_orderd = Time_Orderd
        self.time_order_picked = Time_Order_picked
        self.weather_condiitions = Weather_conditions
        self.road_traffic_density = Road_traffic_density
        self.vehicle_condition = Vehicle_condition
        self.type_of_order = Type_of_order
        self.type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.featival = Festival
        self.city = City


    def get_custom_data_as_df(self):
        
        data_dict = {
            'Delivery_person_Age' : [self.delivery_person_age], 
            'Delivery_person_Ratings' : [self.delivery_person_rating],
            'Order_Date' : [self.order_Date],
            'Time_Orderd' : [self.time_orderd],
            'Time_Order_picked' : [self.time_order_picked],
            'Weather_conditions' : [self.weather_condiitions],
            'Road_traffic_density' : [self.road_traffic_density],
            'Vehicle_condition' : [self.vehicle_condition],
            'Type_of_order' : [self.type_of_order],
            'Type_of_vehicle' : [self.type_of_vehicle],
            'multiple_deliveries' : [self.multiple_deliveries],
            'Festival' : [self.featival],
            'City' : [self.city],
        }
        
        return pd.DataFrame(data_dict)
    