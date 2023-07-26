import tensorflow as tf
import os
import json
from tensorflow import keras

dir = '../Data/Labeled'

Model_location='my_model.h5'

#Load Model from FileSystem
model = tf.keras.models.load_model(Model_location)

#pass the json_data to this method to train the above loaded model
def train(data):
    Train_data=data['raw']
    Train_lable= data['label']
    model.fit(Train_data,Train_lable,epochs=1)

#poll for labled files
#pass each json_file data to the train() method for training
#for root, files in os.walk(dir, topdown=False, onerror=None, followlinks=False ):
#    for name in files:
#        with open(os.path.join(root,name)) as f:
#            data = json.load(f)
#        train(data)

for root, dirs, files in os.walk(dir):
    for name in files:
        with open(os.path.join(root,name)) as f:
            data = json.load(f)
        train(data)
    
model.save(Model_location)
