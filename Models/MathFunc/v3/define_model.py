import tensorflow as tf
import os
import numpy as np
from tensorflow import keras



#Define Model Structure 
# Here we define input structure and Output Structure
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Define the optimizer, loss function and the metrics
model.compile(optimizer='sgd', loss='mean_squared_error')



#Save Model
os.mkdir('saved_model')
model.save('saved_model/my_model.h5') 
