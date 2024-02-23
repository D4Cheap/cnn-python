from flask import Flask

from ml.facial_cnn import FacialCnn

app = Flask(__name__)

@app.route("/predict") 
def predict():
    
    return FacialCnn.train();

@app.route("/train") 
def train():
    
    return FacialCnn.train();