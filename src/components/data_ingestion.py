import os
import sys
import spacy
from src.autologger import logger
from src.exception import CustomException
from src.utils import replace_null
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from dataclasses import dataclass

## intitialize the Data Ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.xlsx')
    test_data_path:str=os.path.join('artifacts','test.xlsx')
    raw_data_path:str=os.path.join('artifacts','raw.xlsx')

## create the data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logger.info('Data Ingestion method starts')

        try:
            logger.info('Dataset read as pandas Dataframe')
            df=pd.read_excel(os.path.join('Data\Womens Clothing Reviews Data.xlsx'))
            
            logger.info("Replacing space from column name & replacing the null values")
            df.columns = df.columns.str.replace(' ', '_')
            df = replace_null(df)
            logger.info("Space from column name & null values replaced")

            logger.info("Taking the X & Y variable and saving it to CLF_data")
            CLF_data = pd.DataFrame(df[['Review_Text','Recommend_Flag']])

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            CLF_data.to_excel(self.ingestion_config.raw_data_path,index=False)

            logger.info('Raw data is created')

            logger.info('Splitting data into train & test')

            train_set,test_set=train_test_split(CLF_data,test_size=0.30,random_state=42)

            train_set.to_excel(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_excel(self.ingestion_config.test_data_path,index=False,header=True)

            logger.info('Data splitted into train & test')

            logger.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            logger.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e,sys)
