from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline



application=Flask(__name__)

app=application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
        if request.method=='GET':
            return render_template('form.html')
        else:
             data=CustomData(
            Review=str(request.form.get('Review'))
        )
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(final_new_data)

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0')