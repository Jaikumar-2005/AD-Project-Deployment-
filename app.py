#!/usr/bin/env python
# coding: utf-8

# In[2]:


# app.py
from flask import Flask,render_template,request
import joblib
import numpy as np


# In[18]:


#Initialize Flask app
app = Flask(__name__)
# load trained model
model = joblib.load('iris_model.pkl')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    try:
        features=[float(request.form[f'feature{i}']) for i in range(1,5)]
    except ValueError:
        return render_template('result.html',prediction="Invalid input. Please enter numeric values.")
    prediction = model.predict([features])[0]
    class_nmae=['Setosa','Versicolor','Virginica']
    result=class_nmae[prediction]
    return render_template('result.html',prediction=result)
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




