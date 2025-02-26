import sys
import os
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#print(sys.path)
# Add src folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from exception import CustomException
from logger import logging

def save_object(file_path, object_name):
    '''
    This function is responsible for saving the object in the file
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            #pickle.dump(object_name, file)
            dill.dump(object_name, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,Y_train,X_test,Y_test,models):
    '''
    This function is responsible for evaluating the models
    '''
    try:
        model_report = {}
        for model_name, model in models.items():
            print(f"Training model starts: {model_name}")
            model.fit(X_train, Y_train) # Train model
            Y_train_pred = model.predict(X_train) # Predict on training data
            Y_test_pred = model.predict(X_test) # Predict on test data
            train_model_score = r2_score(Y_train, Y_train_pred) # Evaluate model on training data
            test_model_score = r2_score(Y_test, Y_test_pred) # Evaluate model on test data
            model_report[model_name] = test_model_score

        return model_report
    except Exception as e:
        raise CustomException(e, sys)