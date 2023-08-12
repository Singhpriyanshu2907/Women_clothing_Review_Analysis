import os
import sys
import yaml
import pickle
import numpy as np 
import pandas as pd
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from textblob import TextBlob
import nltk
from nltk.stem import WordNetLemmatizer
# Initialize NLTK's WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import spacy
from sklearn.svm import LinearSVC
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.autologger import logger



@ensure_annotations
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def replace_null(df):
    # Get a list of columns with the object data type
    object_columns = df.select_dtypes(include='object').columns.tolist()

    # Iterate through each object column and replace null values with mode
    for col in object_columns:
        mode_value = df[col].mode().iloc[0]
        df[col].fillna(mode_value, inplace=True)

    return df


@ensure_annotations
def text_preprocessing(text):
    if isinstance(text, str):
        # Remove leading and trailing whitespaces
        text = text.strip()

        # Convert to lowercase
        text = text.lower()

        # Remove digits and special characters using regular expression
        text = re.sub(r"[-()\"#/@;:{}`+=~|._!?,'0-9]", "", text)

        # Tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)

        # Remove stop words using NLTK
        stop = set(stopwords.words('english'))
        stop1 = set(list(stop)+['always', 'go', 'got', 'could', 'also', 'get', 'us', 'even', 'i', 'm', 'would', 'do', 'go'])
        tokens = [token for token in tokens if token not in stop1]

        # Lemmatize using NLTK's WordNetLemmatizer
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Remove duplicate words
        lemmatized_tokens = list(dict.fromkeys(lemmatized_tokens))

        # Join the tokens back into a cleaned sentence
        cleaned_text = " ".join(lemmatized_tokens)

        return cleaned_text
    else:
        return str(text)

@ensure_annotations
def train_and_predict_models(train_x, train_y, test_x, test_y):

    # Create the model list
    model_list = [
        ('Random Forest', RandomForestClassifier()),
        ('XG Boost', XGBClassifier()),
        ('Extra Trees', ExtraTreeClassifier()),
        ('Linear SVC', LinearSVC()),
        ('KNN', KNeighborsClassifier())
    ]

    results = {}

    for model_name, model in model_list:
        print(f"Training {model_name}...")
        model.fit(train_x, train_y)

        # Make predictions on training and testing data
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        # Calculate accuracy for training and testing data
        train_accuracy = accuracy_score(train_y, train_pred)
        test_accuracy = accuracy_score(test_y, test_pred)

        results[model_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }

    return results