import tensorflow as tf

model = tf.keras.models.load_model('saved_model/my_model.h5')
print('Model Loaded Succesfully')

print(model.predict([30.0]))
