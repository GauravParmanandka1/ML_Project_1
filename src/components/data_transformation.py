import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#print(sys.path)
# Add src folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,numeric_feature_list,categorical_feature_list): # Create all the pickle files for conversion from categorical to numerical data type
        '''
        This function is responsible for data transformation
        '''
        try:
            #numeric_feature_list = ["writing_score", "reading_score"]
            #categorical_feature_list = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            logging.info(f"Numerical columns: {numeric_feature_list}")
            logging.info(f"Categorical columns: {categorical_feature_list}")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))  # Set with_mean=False for sparse matrices
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)) # Set with_mean=False for sparse matrices
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_feature_list),
                    ("cat_pipeline", cat_pipeline, categorical_feature_list)
                ]
            )

            logging.info("Numerical Columns standard scaling completed completed")
            logging.info("Categorical Columns encoding completed")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_data_path:str,test_data_path:str):
        logging.info('Entered Data Transformation component')
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            target_column_name="math_score"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Read the Train and Test Dataset as Dataframe')

            logging.info("Obtaining preprocessing object")

            # Preprocessing the data
            numeric_features = input_feature_train_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = input_feature_train_df.select_dtypes(include=[object]).columns.tolist()
            
            logging.info(f"Applying preprocessing object on training df and testing df")

            preprocessing_obj = self.get_data_transformer_object(numeric_features,categorical_features)

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object_name=preprocessing_obj
            )

            logging.info('Data Transformation completed')

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)