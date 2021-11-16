import keras
import tensorflow as tf

present_model = keras.models.load_model('Weight\diabetes.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(present_model)
tflite_model = converter.convert()

# Save the model.
with open('diabetes.tflite', 'wb') as f:
  f.write(tflite_model)
