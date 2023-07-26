import tensorflow as tf
import os
import json
from tensorflow import keras

#Path of labled files for training
procF = '../Data/Inputs/Processed'

#Model Location
Model_location='../Model/my_model.h5'


#Load Model from FileSystem
model = tf.keras.models.load_model(Model_location)

#pass the json_data to this method to train the above loaded model
def train(data):
    print(data)
    Train_data=data['data']
    Train_lables= data['lables']
    print('Train_data = ' , Train_data,'Train_lables =' , Train_lables)
    model.fit(Train_data,Train_lables,epochs=1)



#Load data from labled and processed files
#pass each json_file data to the train() method for training
for root,dirs,files in os.walk(procF, topdown=False, onerror=None, followlinks=False ):
    for name in files:
        with open(os.path.join(root,name)) as f:
            data = json.load(f)
        train(data)
    
    

model.save(Model_location)


