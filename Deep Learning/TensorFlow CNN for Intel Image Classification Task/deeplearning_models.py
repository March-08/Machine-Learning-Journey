import tensorflow as tf
from tensorflow.python.client import device_lib 
from tensorflow import keras
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D,Flatten,GlobalAvgPool2D,Dense
from tensorflow.python.keras.layers.core import Activation


def convolutional_model(nbr_classes):

    my_input = Input(shape=(150,150,3))
    
    x = Conv2D(32,(3,3),activation="relu")(my_input)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(64,(3,3),activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(units = 32, activation = "relu")(x)
    x = Dense(units = nbr_classes , activation = "softmax")(x)

    return tf.keras.Model(inputs = my_input,outputs = x)

if __name__ =="__main__":
    
    model = convolutional_model(10)
    model.summary()