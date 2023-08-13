import sys
import os
from src.exception import CustomException
from src.autologger import logger
from src.utils import load_object
from src.utils import text_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')

            model=load_object(model_path)

            TFIDF = TfidfVectorizer(analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            encoding='latin-1',
            lowercase = True,
            max_features=1000)

            # Fit the TF-IDF vectorizer to the input text data
            TFIDF.fit(features)


            data_scaled=features.apply(lambda x: text_preprocessing(x))

            transformed_data = TFIDF.transform(data_scaled)

            pred=model.predict(transformed_data)
            return pred
            

        except Exception as e:
            logger.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
           
class CustomData:
    def __init__(self,
                 Review:str
                 ):
        
        self.Review=Review

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Review':[self.Review]
            }

            
            df = pd.DataFrame(custom_data_input_dict)
            logger.info('Dataframe Gathered')
            return df
        except Exception as e:
            logger.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)