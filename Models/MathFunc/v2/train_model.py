import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('saved_model/my_model')
print('Model Loaded Succesfully')

print(model.predict([10.0]))


# Get Training Data
Train_data = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5, 10], dtype=float)
Train_lable = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10, 20], dtype=float)


#Train the Model
model.fit(Train_data,Train_lable,epochs=5)


print(model.predict([10.0]))

model.save('saved_model/my_model')
