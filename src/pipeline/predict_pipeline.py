import sys
import os
import pandas as pd
from dataclasses import dataclass
import dill

#print(sys.path)
# Add src folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models, load_object

@dataclass
class PredictPipelineConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    model_obj_file_path=os.path.join('artifacts', 'model.pkl')

class PredictPipeline:
    def __init__(self):
        logging.info('Entered Predict Pipeline')
        self.predict_pipeline_config = PredictPipelineConfig()
    
    def predict(self,features):
        try:
            logging.info('Entered Predict Function')
            print('Entered Predict Function')
            model=load_object(self.predict_pipeline_config.model_obj_file_path)
            preprocessor=load_object(self.predict_pipeline_config.preprocessor_obj_file_path)
            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)
            print(f"Prediction: {prediction}")
            logging.info(f"For features: {features}")
            logging.info(f"Prediction: {prediction}")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    '''
    This class is responsible for creating the custom data
    '''
    def __init__(self,
    gender: str,
    race_ethnicity: str,
    parental_level_of_education: str,
    lunch: str,
    test_preparation_course: str,
    reading_score: int,
    writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        logging.info('Entered Custom Data')
        print(f"Entered Custom Data:{gender},{race_ethnicity},{parental_level_of_education},{lunch},{test_preparation_course},{reading_score},{writing_score}")
        
    def get_data_as_frame(self):
        '''
        This function is responsible for getting the data as a dataframe
        '''
        try:
            print("In get_data_as_frame function")
            logging.info('Entered get_data_as_frame')
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            custom_data_input_df = pd.DataFrame(custom_data_input_dict)
            logging.info(f'Dataframe created : {custom_data_input_df}')
            print(f'Dataframe created : {custom_data_input_df}')
            return custom_data_input_df
        except Exception as e:
            raise CustomException(e, sys)