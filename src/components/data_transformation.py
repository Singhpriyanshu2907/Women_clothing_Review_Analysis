import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import spacy

from src.utils import save_object
from src.utils import text_preprocessing

from src.exception import CustomException
from src.autologger import logger

from sklearn.feature_extraction.text import TfidfVectorizer



## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_x_file_path=os.path.join('artifacts','preprocessor.pkl')


## Data Ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.DataFrame(pd.read_excel(train_path))
            test_df = pd.DataFrame(pd.read_excel(test_path))

            logger.info('Read train and test data completed')

            logger.info('Dividing the data into x & y variable')
            ## features into independent and dependent features
            train_x = train_df['Review_Text']
            logger.info(f'Shape of train_x: {train_x.shape}')
            train_y = train_df['Recommend_Flag']
            logger.info(f'Shape of train_y: {train_y.shape}')

            test_x = test_df['Review_Text']
            logger.info(f'Shape of test_x: {test_x.shape}')
            test_y = test_df['Recommend_Flag']
            logger.info(f'Shape of test_y: {test_y.shape}')

            logger.info('Applying preprocessing on train x & test x')
            ## apply the transformation on x & y variables
            train_X = train_x.apply(lambda x: text_preprocessing(x))
            logger.info(f'Shape of train_x: {train_X.shape}')
            test_X = test_x.apply(lambda x: text_preprocessing(x))
            logger.info(f'Shape of test_x: {test_X.shape}')

            logger.info('Applying TF-IDF vectorization on train x & test x')
            TFIDF = TfidfVectorizer(analyzer='word',
                             token_pattern=r'\w{1,}',
                             ngram_range=(1, 1),
                             min_df=5,
                             max_df=0.99,
                             encoding='latin-1',
                             lowercase = True,
                             max_features=1000)
            train_x_TFIDF = TFIDF.fit_transform(train_X)
            test_x_TFIDF = TFIDF.transform(test_X)

            logger.info('Getting column names after vectorization')
            train_x_DTM = pd.DataFrame(train_x_TFIDF.toarray(), columns=TFIDF.get_feature_names_out())
            logger.info(f'Shape of train_x_DTM: {train_x_DTM.shape}')
            test_x_DTM = pd.DataFrame(test_x_TFIDF.toarray(), columns=TFIDF.get_feature_names_out())
            logger.info(f'Shape of test_x_DTM: {test_x_DTM.shape}')

            logger.info('Getting x & y variable together for train')
            train_arr = pd.concat([train_x_DTM, pd.DataFrame(train_y, columns=['Recommend_Flag'])], axis=1)
            logger.info(f'Shape of train_arr: {train_arr.shape}')

            logger.info('Getting x & y variable together for test')
            test_arr = pd.concat([test_x_DTM, pd.DataFrame(test_y, columns=['Recommend_Flag'])], axis=1)
            logger.info(f'Shape of test_arr: {test_arr.shape}')

            logger.info('Data transformation completed')


            return(
                train_arr,
                test_arr
            )            


        except Exception as e:
            logger.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)