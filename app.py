import sys
import os
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#print(sys.path)
# Add src folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models
from pipeline.predict_pipeline import PredictPipeline, CustomData

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    logging.info('Entered the home page')
    print('Entered the home page')
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    try:
        if request.method=='GET':
            return render_template('home.html')
        elif request.method=='POST':
            print("Entered the post method")
            data=CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=int(request.form.get("reading_score")),
                writing_score=int(request.form.get("writing_score"))
            )
            print(f"Before getting the data as frame : {data}")
            prediction_df=data.get_data_as_frame()
            print(prediction_df)

            predict_pipeline=PredictPipeline()
            prediction=predict_pipeline.predict(prediction_df)
            print(prediction)
            return render_template('home.html',results=prediction[0])
    except Exception as e:      
        return render_template('error.html',error=e)

if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)