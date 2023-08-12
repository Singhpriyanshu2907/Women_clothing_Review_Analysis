# Basic Import
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.autologger import logger

from src.utils import save_object
from src.utils import train_and_predict_models


from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logger.info('Splitting Dependent and Independent variables from train and test data')
            target_column_name = 'Recommend_Flag'
            X_train = train_array.drop(columns=[target_column_name])
            Y_train = train_array[target_column_name]
            X_test = test_array.drop(columns=[target_column_name])
            Y_test = test_array[target_column_name]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model_list = [
            ('Random Forest', RandomForestClassifier()),
            ('XG Boost', XGBClassifier()),
            ('Extra Trees', ExtraTreeClassifier()),
            ('Linear SVC', LinearSVC()),
            ('KNN', KNeighborsClassifier())
            ]

            model_report = train_and_predict_models(X_train_scaled, Y_train, X_test_scaled, Y_test)
            print(model_report)
            print('\n====================================================================================\n')
            logger.info(f'Model Report : {model_report}')

            # Find the best model based on the highest test accuracy
            best_model_name = max(model_report, key=lambda k: model_report[k]['test_accuracy'])
            best_model_score = model_report[best_model_name]['test_accuracy']

            # Find the model object from the model list based on the best model name
            best_model_object = [model[1] for model in model_list if model[0] == best_model_name][0]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logger.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model_object
            )
          

        except Exception as e:
            logger.info('Exception occured at Model Training')
            raise CustomException(e,sys)