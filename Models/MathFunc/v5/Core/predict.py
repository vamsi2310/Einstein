import tensorflow as tf
import os
import json
from tensorflow import keras
from sklearn import metrics

#Path of labled files for training
uprocF = '../Data/Inputs/UnProcessed'

#Model Location
model_location='../Model/my_model.h5'

#load model from file system
model = tf.keras.models.load_model(model_location)


#Predict using model
def predict():
    prediction = model.predict([30.0])
    print(prediction)
    print(metrics.accuracy_score([30],prediction))
    #print("\n%s: %.2f%%" % (model.metrics_names[1], prediction*100))

predict()
