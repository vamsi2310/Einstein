import tensorflow as tf
import os
import json
from tensorflow import keras
from sklearn import metrics

#Path of labled files for training
Raw_path = '../Data/Raw'
Predictions_path = '../Data/Predictions'

#Model Location
model_location='my_model.h5'

#load model from file system
model = tf.keras.models.load_model(model_location)

#save the pediction
def save_prediction(data,prediction,name):
    file1 = open(Predictions_path+"/"+name,'w')
    file1.write(str(data)+"\n"+str(prediction.tolist())+"\n")
    file1.close()

#Predict using model
def predict(data):
    raw_data=data['raw'] 
    prediction = model.predict(raw_data)
    save_prediction(data,prediction,name)
    #print(metrics.accuracy_score([30],prediction))
    #print("\n%s: %.2f%%" % (model.metrics_names[1], prediction*100))


for root, dirs, files in os.walk(Raw_path):
    for name in files:
        with open(os.path.join(root,name)) as f:
            data=json.load(f)
            prediction = predict(data)
