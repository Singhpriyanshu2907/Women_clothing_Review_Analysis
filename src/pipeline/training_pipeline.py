import os
import sys
from src.autologger import logger
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_set_path,test_set_path=obj.initiate_data_ingestion()
    print(train_set_path,test_set_path)

    data_transformation=DataTransformation()

    train_arr,test_arr=data_transformation.initiate_data_transformation(train_set_path,test_set_path)

    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
