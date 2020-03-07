import tensorflow as tf
from model.blocks import getYoloV3
inputs = tf.keras.Input(shape=(416, 416, 3), name='img')
outputs = getYoloV3(inputs, 6, 100)
model = tf.keras.Model(inputs, outputs, name='darknet53')
model.summary()