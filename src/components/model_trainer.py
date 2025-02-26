import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#print(sys.path)
# Add src folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,Y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            logging.info("Training and Testing data split completed")

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                "XGB Classifier": XGBRegressor(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False), # Verbose set to False to Suppress output
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            ## Hyper parameter tuning
            params = {
                "Random Forest Regressor": {
                    "n_estimators": [8,16,32,64,128,256] #,
                    #"max_depth": 10,
                    #"max_features": ['sqrt', 'log2']
                    },
                "Decision Tree Regressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"] #,
                    #"splitter": ["best", "random"]
                    #"max_depth": 10,
                    #"max_features": ['sqrt', 'log2'],
                    },
                "Gradient Boosting Regressor": {
                    # "loss":['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate":[.1,.01,.05,.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9],
                    # "criterion":['squared_error', 'friedman_mse'],
                    # "max_features":['auto','sqrt','log2'],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "K-Nearest Neighbors Regressor": {
                    #"n_neighbors": 5},
                    # "weights": ['uniform', 'distance'],
                    # "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # "leaf_size": [10, 20, 30, 40, 50],
                    # "p": [1, 2]
                },
                "XGB Classifier": {
                    "learning_rate":[.1,.01,.05,.001],
                    "n_estimators": [8,16,32,64,128,256]
                },
                "CatBoost Classifier": {
                    "depth": [6,8,10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Classifier": {
                    "learning_rate":[.1,.01,0.5,.001],
                    # "loss":['linear','square','exponential'],
                    "n_estimators": [8,16,32,64,128,256]
                }
            }

            model_report,model_report_best_params=evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models, param=params)
            logging.info(f"Model evaluation completed : {model_report}")
            logging.info(f"Model evaluation best params : {model_report_best_params}")
            # To get best model name from dict

            best_model_name = max(model_report, key=model_report.get)
            print(f"Best Model Name : {best_model_name}")
            best_model_score = model_report[best_model_name]
            print(f"Best Model Score : {best_model_score}")
            best_model = models[best_model_name]
            print(f"Best Model : {best_model}")
            if best_model_score < 0.6:
                raise CustomException("Model score is less than 0.6", sys)
            else:
                save_object(file_path=self.model_trainer_config.trained_model_file_path, object_name=best_model)
                logging.info("Model saved successfully")
            
                predicted=best_model.predict(X_test)
                r2_score_value=r2_score(Y_test,predicted)
                logging.info(f"R2 Score : {r2_score_value}")

                logging.info("Model training completed")

                return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)