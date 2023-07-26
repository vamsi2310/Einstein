import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#Define Model Structure
inputs = keras.Input(shape=(1))
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
x= layers.Dense(100,activation="relu")(x)
outputs = layers.Dense(1)(x)


# Here we define input structure and Output Structure
#model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model = keras.Model(inputs=inputs,outputs=outputs,name="Math_Func")
#Define the optimizer, loss function and the metrics
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[tf.keras.metrics.Accuracy()])

#Save Model
model.save('my_model.h5')
model.summary()
