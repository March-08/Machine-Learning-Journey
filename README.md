# Machine-Learning-Journey
This repository contains projects in the field of machine learning and deep learning that I have developed and that I continue to develop on my path to becoming a professional ML engineer. It includes links to the source codes that I have published as well as other related activities such as articles that I have published in journals like [Towards Data Science](https://towardsdatascience.com/search?q=marcello%20politi) and [Geek Culture](https://medium.com/geekculture/tagged/tensorflow).
My knowledge comes from work experiences like the internship at [INRIA](https://www.inria.fr/en/inria-centre-universite-cote-dazur) where for my thesis project I investigated [pruning methods](https://arxiv.org/pdf/2003.03033.pdf) for neural network compression in [Julia](https://julialang.org/). Moreover given my personal passion for these topics I have studied independently on several books such as : 
- [Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.amazon.it/Hands-Machine-Learning-Scikit-learn-Tensorflow/dp/1492032646) by Aurelien Geron,
- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.amazon.it/Practical-Statistics-Data-Scientists-Essential/dp/149207294X/ref=asc_df_149207294X/?tag=googshopit-21&linkCode=df0&hvadid=459278696633&hvpos=&hvnetw=g&hvrand=13953028916279784642&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1008736&hvtargid=pla-913779307193&psc=1) by Peter Gedeck
- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville.
 
I also regularly consume machine learning related material such as the following youtube channels :
- [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)
- [Keen Jee](https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg)
- [standford online](https://www.youtube.com/channel/UCBa5G_ESCn8Yd4vw5U-gIcg)
- [freeCodeCamp](https://www.youtube.com/channel/UC8butISFwT-Wl7EV0hUK0BQ)
- [Tech with Tim](https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg)
- [MIT open course Ware](https://www.youtube.com/c/mitocw)
 
Finally in my learning path I have to mention the most common platforms such as [Udemy](https://www.udemy.com/) where I enrolled a course for the implementation of deep learning algorithms on mobile devices called [Deep Learning Course with Flutter & Python](https://www.udemy.com/course/flutter-deeplearning-course/), and [coursera](https://www.coursera.org/) where I learned the basics from the courses of [Andrew NG](https://www.coursera.org/specializations/deep-learning) 

I also starte studying low level programming using cuda in order to boost deep learning performances, most of the framework such as TensorFlow and Pytorch are based on kernel lauches, check ouut my [**Cuda Programming Repo**](https://github.com/March-08/CUDA-programming).

While in my [**A.I Art**](https://github.com/March-08/GANs-A.I-Art-) repository I have started publishing scripts about GANs that are able to generate art in terms of picture, audio, text etc...

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
  
  
- [Dimensionality Reduction](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/Dimensionality_Reduction.ipynb)
   
  Projection and Manifold Learning. PCA, Kernel PCA, Incremental PCA to speed up computation and for data visualization.

- [Solution to Titanic-Machine Learning from Disaster Kaggle's challenge](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/Titanic/Titanic_Marcello_Politi.ipynb)  
    
  Kaggle challenge : https://www.kaggle.com/c/titanic/overview  
  In this code I developed a custom sklearn tranformer.   
  I used a pipeline to preprocess the data.   
  The model selection was run using both randomized_search and optuna.  
  

- [Basic operations in PyTorch](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/Basic_of_PyTorch.ipynb)  
    
- [Linear and Logistic Regression in PyTorch](https://github.com/March-08/Machine-Learning-Journey/blob/main/Machine%20Learning/Basic_of_PyTorch.ipynb)  
  
## Deep Learning
 - [Introduction to Neural Networks with Tensorflow](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow)
     
    The scripts in this directory are inspired by the book Hands-On Machine LEarning with Scikit-Learn, Keras & Tensorflow.
 I read about the origin of deep learning and how it was inspired by biological neural networks, starting from [perceptron](https://en.wikipedia.org/wiki/Perceptron).
As a first step, I saw how the [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm is heavily used in deep learning. I then learned what [automatic differentiation](https://www.tensorflow.org/guide/autodiff) is, which is the basis of backpropagation. I also saw how to derive functions using tensorflow's [Gradient Tape](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow/GradientTape.ipynb), which records various operations and allows you to compute derivatives efficiently.
I learned how neural networks can be applied to both classification and regression tasks.
The implementation of such networks can be more or less difficult depending on the API used. The [sequential API](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow/Regression_Classification_using_Sequential_API.ipynb) allows only a sequential implementation of the layers, so it loses generality but it is very simple. The [Functional API](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow/Complex_Models_with_Functional_API.ipynb) instead allows to create more complex structures like [wide and deep model for reccomendation systems](https://arxiv.org/abs/1606.07792). The last method using the [subclassing API](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow/Flexibility_with_Subclassing_API.ipynb) allows a total freedom in the architecture and functionality of the network, usually more used in research.
I also learned useful utils like [saving and loading models, using callbacks and creating custom callbacks, and monitoring the training progress using the tensorboard](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow/Saving_Callbacks_TensorBoard.ipynb).
Finally as I have already done for ML algorithms in sklearn, I learned that it is possible to [wrap a deep learning model in a sklearn regressor](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow/Fine_Tuninng_NN_Parameters.ipynb) and run a random-search or grid-search to make model selection.
 
- [Intro to Computer Vision](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/Intro%20to%20Computer%20Vision)
     
    How Convolution and Pooling layers works. How you can use these layers to extract features from pictures and reduce the size of it augmenti the depth (number of filters).
 Here I started workin on computer vision using the **Fashion Mnist** dataset provided by the keras **APIs**.
  
   Training on a custom dataset using [ImageDataGenerator](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Intro%20to%20Computer%20Vision/Training_with_ImageDataGenerator.ipynb) and the most common flow_from_directory function. useful to preprocess the data, rescaling, crop the size etc..
   The steps_per_epoch is an important factor during the training step, because a generator can potentially generate an infinite number of images.
  
 
   [Text Generation from Shakespeare Data]( https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Intro%20to%20Natural%20Language%20Processing/Text_Generation_from_Shakespeare_Data.ipynb)

   In [Mnist Sign Language project](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Intro%20to%20Computer%20Vision/Sign_Language_Classification.ipynb), I create a data generator starting from a csv file decoded by pandas. I split the train val using the stratify option of pandas to mantain the distribution, and I have applied data augmentation for a robust classifier
 
  In the [Dogs vs Cats project](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Intro%20to%20Computer%20Vision/Dogs_vs_Cats.ipynb), I use the data augmentation approach, and analyzed the differences with a previous baseline. Keras allows to implement this powerful methodology using few lines of code.
 
- [Intro to Natural Language Processing](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/Intro%20to%20Natural%20Language%20Processing)
    
  I staterd [here](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Intro%20to%20Natural%20Language%20Processing/Tokenize%20words.ipynb) learning about the tokenizer of tensorflow , and how you can tokenize sentences and also add padding to match the size of a neural network.
    
    [This project](https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Intro%20to%20Natural%20Language%20Processing/Tokenize%20words.ipynbhttps://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/Intro%20to%20Natural%20Language%20Processing/Tweet_Sentiment_Classification_using_Pre_Trained_Embedded.ipynb) is about classifying tweets based on their sentiment. I have used a transfer learning approach, using a pre-trained embedding provided by [Standford](https://nlp.stanford.edu/projects/glove/). I used RNN, in particular both LSTM and GRU to see their differences.

 
- [Self Normalize Network](https://github.com/March-08/Machine-Learning-Journey/tree/main/Deep%20Learning/Introduction%20to%20Neural%20Networks%20with%20Tensorflow)
     
    In the paper https://arxiv.org/pdf/1706.02515.pdf, authors showed that a Forward NN with SELU activation functions in able of self normalizing (mean 0 , var 1 after each layer), so it can resolv the vanishing/exploding gradient problem (no need of batch normalization). However few conditions are needed:

   - Inpute features must be standardized
   - LeCun normal inizialization
   - Sequential architecture
   - All layers Dense
 
   They also propose to use AplphaDroput in order to mantain same mean and variance after each dense layer
 
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
       
     <img align="left" src="https://github.com/March-08/Machine-Learning-Journey/blob/main/Deep%20Learning/FlowerDetector/flower_detector.jpeg" height="290">
    Development of CNN for flower detection in Tensorflow. Implementation of flutter application using the exported network adopting the tflite package. You can find the dataset at the following link : https://www.tensorflow.org/datasets/catalog/tf_flowers). The user is able to take a picture or upload a picture from is gallery, and the app detects if the flowers is among these categories : daisy, dandelion, roses, sunflowers and tulips
        
    
