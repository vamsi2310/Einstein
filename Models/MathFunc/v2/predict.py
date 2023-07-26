import tensorflow as tf

model = tf.keras.models.load_model('saved_model/my_model')
print('Model Loaded Succesfully')

print(model.predict([230.0]))
