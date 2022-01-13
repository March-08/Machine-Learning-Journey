from scipy.sparse import base
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense,Input,GlobalAveragePooling2D
from tensorflow.python.keras import activations
from utils import get_avg_size,split_data
import os

avg_sizes = (397,364,3)

base_model = tf.keras.applications.MobileNetV2(input_shape=avg_sizes,include_top=False,weights='imagenet')
base_model.trainable = False


def mobileNet_based_model():
    input_layer = Input(shape=avg_sizes),
    x = base_model(input_layer,training = False)
    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units = 64,activation = "relu")(x)
    x = Dense(units = 32,activation = "relu")(x)
    x = Dense(units = 1, activation = "sigmoid")(x)

    return tf.keras.Model(inputs = input_layer, outputs = x)


if __name__ == "__main__":

    model = mobileNet_based_model()
    model.summary()