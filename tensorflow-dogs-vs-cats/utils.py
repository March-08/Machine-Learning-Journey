import os
import shutil 
import glob

from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DOG = "1"
CAT = "0"

avg_size = (397,364)

def download_data():

    os.system("pip install -q kaggle")
    os.system("mkdir ~/.kaggle")
    os.system("mv kaggle.json /root/.kaggle/")
    os.system("chmod 600 /root/.kaggle/kaggle.json")
    os.system("kaggle competitions download -c 'dogs-vs-cats' -p ./data")

    os.system("makdir ./data")
    os.system("mkdir ./data/training")
    os.system("mkdir ./data/training/val")
    os.system("mkdir ./data/training/train")
    os.system("mkdir ./data/test")
    os.system("mkdir ./data/test/images")

    os.system("unzip -j ./data/test1.zip -d ./data/test/images")
    os.system("unzip -j ./data/train.zip -d ./data/training")



def split_data(path_to_training , split_size = 0.2):

    images_paths = glob.glob(os.path.join(path_to_training,"*.jpg"))

    train_set , val_set = train_test_split(images_paths, test_size = split_size)

    path_to_dog_validation = os.path.join(path_to_training,"val/" + DOG)
    path_to_cat_validation = os.path.join(path_to_training,"val/" + CAT)

    if not os.path.isdir(path_to_dog_validation):
        os.makedirs(path_to_dog_validation)

    if not os.path.isdir(path_to_cat_validation):
        os.makedirs(path_to_cat_validation)


    path_to_dog_training = os.path.join(path_to_training,"train/" + DOG)
    path_to_cat_training = os.path.join(path_to_training,"train/" + CAT)
    

    if not os.path.isdir(path_to_dog_training):
        os.makedirs(path_to_dog_training)

    if not os.path.isdir(path_to_cat_training):
        os.makedirs(path_to_cat_training)


    for x in train_set:
        basename = os.path.basename(x)
        if basename.split(".")[0] == "cat":
            shutil.move(x,path_to_cat_training)
        else:
            shutil.move(x,path_to_dog_training)

    for x in val_set:
        basename = os.path.basename(x)
        if basename.split(".")[0] == "cat":
            shutil.move(x,path_to_cat_validation)
        else:
            shutil.move(x,path_to_dog_validation)



#get avg sizes of images to use as input of the model
def get_avg_size(path_to_training):
    width = 0 
    height = 0
    n = 0

    path_to_dog_folder = os.path.join(path_to_training,"train/" + DOG)

    for filename in os.listdir(path_to_dog_folder):
        n = n+1
        with Image.open(os.path.join(path_to_dog_folder, filename)) as image:
            width = width + image.size[0]
            height = height + image.size[1]
            
    print(width//n,height//n)
    return (width//n,height//n)
   

def test_generator(batch_size , path_to_test):

    testing_preprocessor = ImageDataGenerator(
        rescale = 1/255.
       )

    test_generator = testing_preprocessor.flow_from_directory(
        path_to_test,
        class_mode="binary",
        target_size= avg_size,
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
        )

    return test_generator


def train_val_generators(batch_size, path_to_training):
     
    path_to_train = os.path.join(path_to_training, "train")
    path_to_val = os.path.join(path_to_training, "val")

    training_preprocessor = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

    validation_preprocessor = ImageDataGenerator(
        rescale = 1/255.
       )


    train_generator = training_preprocessor.flow_from_directory(
        path_to_train,
        class_mode="binary",
        target_size= avg_size,
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = validation_preprocessor.flow_from_directory(
        path_to_val,
        class_mode="binary",
        target_size= avg_size,
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )


    return train_generator, val_generator


if __name__ == "__main__":

    path_to_training = "./data/training"

    download_data()
    split_data(path_to_training , split_size = 0.2)
    get_avg_size(path_to_training) #(397,364)

    
