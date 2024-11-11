from flask import Flask

from ml.facial_cnn import FacialCnn

app = Flask(__name__)

#https://pythonbasics.org/flask-tutorial-routes/
#https://realpython.com/python-profiling/
@app.route("/predict/image/<directory>/<image_file>") 
def predict(directory, image_file):
    
    return FacialCnn.predict(directory, image_file);

@app.route("/train") 
def train():
    history = FacialCnn.train()


    return "###### Total Time Taken: " + str(history['time']) + ' seconds ######\n' +  "###### Accuracy: " + str(history['accuracy']);

@app.route("/train/stress") 
def train_stress():
    total_time = 0
    total_acc = 0
    runs = 30

    for _ in range(runs):
        result = FacialCnn.train()
        total_time += result["time"]
        total_acc += result["accuracy"]
    
    average_time = total_time / runs
    average_accuracy = total_acc / runs

    return "###### Average time to train: " + str(average_time) + ' seconds ######\n' +  "###### Average accuracy of the model: " + str(average_accuracy);