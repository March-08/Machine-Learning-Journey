import tensorflow as tf
import os 

from tensorflow.python.ops.gen_batch_ops import batch
from deeplearning_models import mobileNet_based_model
from utils import create_generators

from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.eager.monitoring import Metric


path_to_training = "./data/training"
path_to_test="./data/test"


path_to_train = os.path.join(path_to_training,"train") 
path_to_val= os.path.join(path_to_training,"val") 

train,val,test = create_generators(batch_size = 64 , path_to_training = path_to_training , path_to_test = path_to_test)

TRAIN = True
TEST = False

if TRAIN :
    epochs = 8
    batch_size = 64

    path_to_save_model = "./Models"

    ckpt_saver= ModelCheckpoint(
        filepath = path_to_save_model,
        verbose = 1,
        monitor = "val_loss",
        save_best_only=True,
        save_freq="epoch",
    )


    early_stop = EarlyStopping(
        monitor = "val_loss",
        patience= 2,
        restore_best_weights=True
    )

    model = mobileNet_based_model()

    model.compile(
        optimizer="adam",
        loss = "binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train,
        validation_data=val,
        batch_size = batch_size,
        epochs=epochs,
        callbacks = [early_stop,ckpt_saver]
    )

