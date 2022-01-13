# Machine-Learning-Journey
This repository contains projects in the field of machine learning and deep learning that I have developed and that I continue to develop on my path to becoming a professional data scientist. It includes links to the source codes that I have published as well as other related activities such as articles that I have published in journals like Towards Data Science and Geek Culture.


## Machine Learning

- [Linear Regression and GD from scratch with numpy](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/multi_linear_regession.ipynb)

<a target="_blank" href="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/1"><img src="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/1" alt="Recent Article 1"> 
  
 - [Binary Classification using scikit-learn](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/Binary_Classification.ipynb)
     
     I used the mnist dataset, modifying it slightly so that I could utilize a classifier that could recognize if a digit was "5" or "not 5".
I evaluated the algorithms involved, random forest, SGD, and a dummy algorithm, using cross validation to be as accurate as possible.
I noticed that in this case using the metric "accuracy" was not of much help, in fact even the dummy algorithm had a very high accuracy. This is because there are many more "not 5" images than "5" images.
So I delved into metrics like Precision Recall and F1.
I used these metrics to plot a PR curve comparing it also with various types of thresholds to understand which was the optimal point of the threshold to make classification.
I also delved into the ROC curve and the AUC area.
I used the mentioned ones to compare the various algorithms and understand that the best was the random forest.
  
- [Ensemble Learning: Bagging, Pasting, Boosting and Stacking](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/Ensemble_Learning.ipynb)
     
     **Ensemble learning** : combine few good predictors (decision tree, svm, etc...) to get one more accurate one predctor.
   
  **Bagging and Pasting**:
   
    These approaches use the same training algorithm for every predictor, but train them on different subsets of the training set. When sampling is performed with replacement, the method is called bagging, pasting otherwise. Random Forest is a example of bagging using decision trees, one of the most powerful algorithm in ML. It can also be used for feature selection.
   
  **Out of bag evaluation**
Some instances may be sampled several times during bootstrapping, while others may not be sampled at all, these are called out-of-bag instances.
   
  **Boosting** This is another ensemble solution. The most famous boost methods are AdaBoosting and Gradient Boost.
   
  **AdaBoost**: Each new predictor (model) in the esnemble should focus on correct the instances that its predecessor underfitted, weighting the missclassified instances. The boosting cannot be parallelized, because each predictor should wait for the previous one. In scikit learn the "SAMME" algorithm is used for multiclass labels AdaBoost. While "SAMME.R" relies on probabilities instead of predictions, usually performs better.
   
  **GradientBoost**: Similar to AdaBoosting but instead of working on the weights, each predictor tries to fit the residuals errors of the previous predictor.
   
  **Stacking**: This is the last ensemble method. Instead of aggregating the predictors with trivial methods like majority voting, we train a model to perform the aggregation. Each tree predicts a certain value, and the final predictor called blender or meta-learner takes these predictions and output the final value.
  
  
  


- [Solution to Titanic-Machine Learning from Disaster Kaggle's challenge](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/Titanic/Titanic_Marcello_Politi.ipynb)  
    
  Kaggle challenge : https://www.kaggle.com/c/titanic/overview  
  In this code I developed a custom sklearn tranformer.   
  I used a pipeline to preprocess the data.   
  The model selection was run using both randomized_search and optuna.  
  
  
## Deep Learning
 
- [CNN for pneumonia classification from chest X Rays images](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/Pneumonia-Chest-X-Rays-Classifier/Pneumonia-Chest-X-Rays-Classifier-main)
     
    The dataset provided by Kaggle and available at https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images and 2 categories (Pneumonia/Normal). The Pneumonia folder contains images of two different categories virus and bacteria, so with the additional normal category we have 3 different classes. 
 
- [Transfer Learning for Robust Image Classification](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/tensorflow-dogs-vs-cats)
  
  <a target="_blank" href="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/2"><img src="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/2" alt="Recent Article 2"> 
  
    
- [TensorFlow CNN for Intel Image Classification Task](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/TensorFlow%20CNN%20for%20Intel%20Image%20Classification%20Task)
 
    <a target="_blank" href="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/3"><img src="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/3" alt="Recent Article 3"> 
      
- [Iterative Pruning Article using Julia](https://towardsdatascience.com/iterative-pruning-methods-for-artificial-neural-networks-in-julia-c605f547a485)
 
    <a target="_blank" href="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/5"><img src="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/5" alt="Recent Article 5"> 
       
 - [Best System Award EVALITA 2020 : Stance Detection System](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/Stance_Detection_Evalita2020/Stance_Detection-master)
         
    EVALITA is a periodic evaluation campaign of Natural Language Processing (NLP) and speech tools for the Italian language, born in 2007.
    This year EVALITA provided a task called [SardiStance](http://www.di.unito.it/~tutreeb/sardistance-evalita2020/index.html), that is basically a stance detection task,         using a data-set containing Italian tweets about the [Sardines Movement](https://en.wikipedia.org/wiki/Sardines_movement).
    The task is a three-class classification task where the system has to predict whether a tweet is in favour, against or neutral/none towards the given target,                 exploiting only textual information, i.e. the text of the tweet.
    The dataset will include short documents taken from Twitter.
    The evaluation will be performed according to the standard metrics known in literature (accuracy, precision, recall and F1-score)

- [The remaking of the Silicon Valley’s series SeeFood App](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/seeFood%20App)
 
    <a target="_blank" href="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/0"><img src="https://github-readme-medium-recent-article.vercel.app/medium/@marcellopoliti/0" alt="Recent Article 0"> 

 - [Flower Detector mobile app](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/FlowerDetector)
       
     <img align="left" src="https://github.com/March-08/Machine-Learning-Journey/blob/main/FlowerDetector/flower_detector.jpeg" height="290">
    Development of CNN for flower detection in Tensorflow. Implementation of flutter application using the exported network adopting the tflite package. You can find the dataset at the following link : https://www.tensorflow.org/datasets/catalog/tf_flowers). The user is able to take a picture or upload a picture from is gallery, and the app detects if the flowers is among these categories : daisy, dandelion, roses, sunflowers and tulips
        
    
