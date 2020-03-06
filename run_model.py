import tensorflow as tf
from model.blocks import getDarkNet53
inputs = tf.keras.Input(shape=(416, 416, 3), name='img')
outputs = getDarkNet53(inputs)
model = tf.keras.Model(inputs, outputs, name='darknet53')
model.summary()