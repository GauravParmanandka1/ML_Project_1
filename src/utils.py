import sys
import os
import pandas as pd
import numpy as np
import dill

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