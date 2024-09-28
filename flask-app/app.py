from flask import Flask,render_template,request
import mlflow
from preprocessing_utility import normalize_text
import pickle

import dagshub

mlflow.set_tracking_uri('https://dagshub.com/sandeeprairai/mlops_mini_project2.mlflow')
dagshub.init(repo_owner='sandeeprairai', repo_name='mlops_mini_project2', mlflow=True)



app=Flask(__name__)

# load  the model
model_name="my_model"
model_version=2

model_uri=f'models:/{model_name}/{model_version}'
model=mlflow.pyfunc.load_model(model_uri)

vectorizer=pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']


    #load the model fro model registry

    #clean
    text=normalize_text(text)

    # bow
    features=vectorizer.transform([text])

    #prediction
    result=model.predict(features)

    #show
    return render_template('index.html',result=result[0])

app.run(debug=True)