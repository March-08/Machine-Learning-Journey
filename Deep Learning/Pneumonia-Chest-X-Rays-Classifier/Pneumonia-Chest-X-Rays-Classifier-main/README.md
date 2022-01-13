# Pneumonia-Chest-X-Rays-Classifier
Convolutional Neural Network for pneumonia classification from chest X Rays images.

# Dataset
The dataset provided by Kaggle and available at https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images and 2 categories (Pneumonia/Normal).
The Pneumonia folder contains images of two different categories **virus** and **bacteria**, so with the additional **normal** category we have 3 different classes.
Here it is an example for the three classes:

<div align="center">
    <img src="https://github.com/March-08/Pneumonia-Chest-X-Rays-Classifier/blob/main/Chest.PNG" width="800px"</img> 
</div>

The dataset has the following distribution:

<div align="center">
    <img src="https://github.com/March-08/Pneumonia-Chest-X-Rays-Classifier/blob/main/distribution.PNG" width="400px"</img> 
</div>


# The model
I used a CNN with the following architecture:

<div align="center">
    <img src="https://github.com/March-08/Pneumonia-Chest-X-Rays-Classifier/blob/main/architecture.PNG" width="400px"</img> 
</div>




# Hyperparameters
- Batch size = 32
- Epochs = 8
- Loss function = categorical_crossentropy
- Optimizer= adam

# Train step
Here a report about the training step, I focused for this project only on the accuracy measure.
<div align="center">
    <img src="https://github.com/March-08/Pneumonia-Chest-X-Rays-Classifier/blob/main/Train.PNG" width="400px"</img> 
</div>

# Test Accuracy
<div align="center">
    <img src="https://github.com/March-08/Pneumonia-Chest-X-Rays-Classifier/blob/main/Test.PNG" width="400px"</img> 
</div>





