# sentiment-analysis
1. Abstract
 
Sentiment Analysis is also called opinion mining as it alludes to the utilization of natural language processing, computational linguistics and text analysis to discover how people feel about a topic. It determines whether our subject is positive, negative or neutral. Sentiment analysis approaches are grouped into three main categories – statistical, knowledge-based and hybrid techniques. Statistical methods are based on machine learning approaches like latent semantic analysis. Knowledge-based methods classify text based on words like happy, sad, bored etc. Hybrid methods combine both the approaches.
Deep Learning belongs to a broader family of machine learning methods. It uses a cascade of many layers of nonlinear processing units for feature extraction and transformation. Deep Learning is based on “representation learning” of data. Representation learning is a set of techniques that transforms a raw data input to a representation that can be effectively exploited by machine learning tasks.
Our application will collect reviews about movies from various online resources and perform sentiment analysis on the collected information using deep learning. The output of our deep learning algorithm will determine if the movie had a positive, negative or neutral effect on the audience. 

2. Architecture

	Learning algorithms can be divided into supervised and unsupervised learning methods.
In supervised learning, the control knows the correct answer for the set of selected inputs. But in unsupervised learning methods, the answer is not known beforehand.
  ![image](https://user-images.githubusercontent.com/26388650/43681184-129007ae-9802-11e8-8567-e61f8b606a18.png)


2.1. Perceptron Learning Algorithm (PLA) 
PLA belongs to the class of supervised learning methods. It is a binary classifier and decides if the input  belongs to one set of results or another. The architecture of a PLA model is as follows - 
	

 ![image](https://user-images.githubusercontent.com/26388650/43681219-9eb41fd6-9802-11e8-90f8-8c9a1895896f.png)


	

2.2. Convolutional Neural Network (CNN) 
	Convolutional Neural Network is a type of feed-forward artificial neural network. It is biologically-inspired variant of Multi-Layer-Perceptrons. 


Its architecture is as follows -
 
![image](https://user-images.githubusercontent.com/26388650/43681222-b5e551e8-9802-11e8-872d-bcb918500a35.png)


To understand convolutional neural network, we must understand four main concepts regarding convolutional neural network
Convolution
Non-linearity
Pooling
Classification

Convolution:
The main reason for using a convolutional network is because they were found to be very effective for image recognition, character recognition, object recognition and many more. The convolutional neural network gets its name from the convolution operator.  The main purpose of convolutional layer is to extract all the features from the input. Convolutional neural network is made to understand the image input, understand its spatial structure. Same principle is now applied on a sequence of words. It has the ability to learn the structure of the paragraphs in the input. The output of this layer is a feature map.

Non-linearity:
Relu:
![image](https://user-images.githubusercontent.com/26388650/43681227-c76dcec2-9802-11e8-8344-beb237492ebd.png)

Relu is a non-linear operation. It means rectified linear unit. Relu function is defined as f=max(zero,input) . The real-world input data is nonlinear, and as we know convolution is linear, we introduce relu to introduce nonlinearity into the model. Relu does element-wise operation. The feature map that we get as output is termed as rectified feature map.
 
 This is a Rectified Linear Unit (ReLU) activation function, which gives zero for x < 0 and then linear with slope 1 when x > 0
Sigmoid:

Sigmoid is an activation function which when given input and puts it in the range between 0 and 1. The sigmoid function is termed as  σ(x)=1/(1+e^−x) . 
 ![image](https://user-images.githubusercontent.com/26388650/43681232-d9092622-9802-11e8-88ff-7ca45c92aead.png)

 			Fig - Sigmoid Function


Pooling:
Pooling is also called as down sampling or subsampling. It helps in reducing the reducing the input feature map dimensionality while making sure that no important information is lost. We have different types of pooling like sum pooling, average pooling and max pooling. Max pooling is one of the most commonly used. Pooling makes the input manageable. It also helps in controlling overfitting. We may have many pooling layers in the convolutional neural network. 

Fully connected Layer:
This layer is the last layer of the convolutional neural network. It is used for classification. Thus we can say that convolution and non-linearity helps in feature engineering while pooling and fully connected layers act as classifiers. The fully connected layer is a multilayer perceptron with an activation function. The reason for fully connected layer is to classify the input into various classes. In our project, we are doing binary classification i.e., positive or negative.

The architecture that we are using in our convolutional neural network for sentimental analysis
First layer is the embedding layer.
Second layer is the convolution layer.
Third layer is the pooling layer.
Last layer is the fully connected layer.

2.3 Software Used 

1.	Anaconda
2.	Tensorflow
3.	Keras 

1.Anaconda:

	Anaconda is free package manager, environment manager, python distribution many with inbuilt packages. It consist of more than 720 packages which can be used for several applications like interactive data visualization, machine learning, deep learning and Big Data. 
Anaconda gives easy access to deep learning packages like theano, tensor flow, neon, h2o, keras and many more. Any anaconda packages can be installed, removed and updated. Anaconda comes with jupyter notebook extensions . we have used jupyter notebook packages n our project.

2.TensorFlow:

	TensorFlow is open source software library developed by google for numerical computation using dataflow graphs . Nodes in the graph represents mathematical operations of nodes and edges of the graph represent tensors. TensorFlow has flexible architecture which allows computations for more than one CPUs or GPUs  in a desktop, server, or mobile devices with single API.

Tensor is a central unit of the data . tensor consist of set of values into an array of any dimension. For example,
[4. , 7. , 8.]  # rank 1 tensor   . 
[[4. , 7. , 8.] , [5. , 2. , 7.] ] rank 2 tensor . 

Rank of the tensor represents dimension.
TensorFlow can be imported by adding below statement in jupiter notebook.

import tensorflow as tf 

3.Keras:       
Keras is a neural network based API written developed for python . keras is capable of running on the top of either TensoFlow or Theano library. Keras can be used for both convolutional networks and recurrent networks. 
Guiding Principle:
User Friendliness: Keras is an API designed for human beings. It offers consistent & simple APIs. it minimizes the number of user action required for common use cases , and it provides clear and actionable feedback upon user error. 
Modularity: A model is understood as a sequence or a graph of standalone, fully configurable modules that can be plugged together with very few restrictions.
Extensibility:Easy to add new modules in keras. keras provides many examples for existing modules as well. This features of keras are very useful in case of advance research. 
In our project, we have used keras sequential model which is stack of layers. It can be created by passing a list of layers instances to the constructor. for example , 
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
       Dense(200),
    Activation('softmax'),
])
Or it can be added by using .add() method. 
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
The input shape argument is a shape tuple of integers. For 2D layers, dense uses  input_dim  argument for specifying input shape and for 3D layers it uses input_dim and input_lenth.

For training a model , we need to first configure the learning process which is done by using compile method. For example ,

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Optimiser argument is required for compiling keras model . we have used adam optimizer in our project. it is useful for stochastic gradient . it is generally used for problems which has large data set or many parameters.  Loss function  is an objective that model will try to minimize. We  have used  existing loss function categorical_crossentropy in our project.  The metrics function used to judge the performance of the model .
3. Dataset Used 

	In this project we have used Internet Movie Database (IMDB) for movie reviews. It is an online database of information related to films, video games and  television programs including cast, production crew, summaries, fictional characters, plot and reviews, operated by IMDb.com.Keras provides an in-built access to IMDB dataset. Large Movie Review dataset has 25,000 reviews for training and testing. These reviews are classified as good or bad. The words in the reviews are converted into integers that determine the polarity of the word in the dataset. Hence, each review sentence consists of a sequence of integers.
The data set consist of total 50000 IMDB reviews which is  divided in two equal parts for training and testing. The sentiments in reviews is  in binary. For ratings less than 5, sentiment score is 0 and for ratings greater than  or equal to 7, sentiment score is 1. All this Reviews have been pre-processed. each review is encoded as a sequence of word indexes. words are indexed by overall frequency in the dataset. For example, word with an index "1" represents  the most frequent word in the data or a word indexed as “5” would show 5th most used word. This feature would be helpful for filtering operation.
4. Feature Engineering Tasks 
	
	In our project we have designed two models - PLA and neural network model. Both the models use IMDB dataset as input and generate the output. This output is compared with the already calculated results of the IMDB dataset and the accuracy of each model is measured based on how close the results are to the actual results.

We have used keras and tensorflow in jupyter notebook. Keras provides an in-built function to access the movie review dataset of IMDB. The following command is used to import the IMDB dataset - 
import numpy from keras.datasets import imdb 

After the dataset is imported, we load the dataset using the load_data() function. To make the data in the dataset available to our models designed, we use the following command - keras.datasets.imdb.load_data().
When we call the load_data() function for the first time on the dataset, the IMDB dataset is first stored in our local at - ~/.keras/datasets/imdb.pkl as a 32 megabyte file. 
The load_data() function takes the number of words to load as a parameter. This number should not be greater than 25,000 as it is the maximum limit. In our models, this number is 4000 (imdb.load_data(4000)). These words are the top 4000 words of the IMDB movie review dataset.
Next we create the models and feed the top 4000 words of IMDB movie review dataset as input. The inputs to both the layers is an embedding layer with a vocabulary of 4000, 32 dimensional word vector size and input length of 400 words.

We have checked the number of unique words in the dataset

![image](https://user-images.githubusercontent.com/26388650/43681238-efebd7b8-9802-11e8-8675-876ac2fb8c2a.png)

 
To get some idea of the review we have calculated the mean review length and obtained the following results
 
![image](https://user-images.githubusercontent.com/26388650/43681243-07ab5040-9803-11e8-8891-4999cd466df7.png)


5. Model Evaluation 
	5.1. Multi-Layer Perceptron Model 
		The multi-layer perceptron model designed in our project is a sequential model. It takes an embedding layer with a vocabulary of 4000, 32 dimensional word vector size and input length of 400 words as input. This embedding layer gives a 32X400 matrix as output. This matrix is converted to one dimension using the Flatten() function - model.add(Flatten()) . 
Requirements for multi-layer perceptron model:
 

The outputs learned from each layer:
 

Fitting the model:
	5.2 Convolutional Neural Network
 Keras has the functionality to support convolutional and maxpooling. The classes that we imported are Convolution1D and MaxPooling1D. Below is the snippet of all the classes imported.
 ![image](https://user-images.githubusercontent.com/26388650/43681246-1377b0a8-9803-11e8-9971-6e4db104db6d.png)


 We have taken only the top 4000 words and for reviews we have taken only 300 words for each review. After the initial word embedding layer we use the convolution layer.
 
Convolution layer: 
This convolutional layer has 32 feature maps and reads embedded word representations 3 vector elements of the word embedding at a time. 
pooling:
The convolutional layer is followed by a 1D max pooling layer with a length and stride of 2 that halves the size of the feature maps from the convolutional layer. 
We have used adam optimizer to calculate the gradient and have done back propagation to update weights. 

The outputs learned from each layer:
 

Model fitting:
We have fit the model using the x_train, y_train ,x_test and y_test. Model evaluation is done on the testing dataset and accuracy is calculated. 
Epoch: epoch is used for iterative algorithms like a neural network. Epoch is defined as one single round or iteration over the training set. 
 
6. Challenges Faced and techniques used to overcome
•	Setting up the environment was time consuming.
•	Initially, we went ahead and used the imdb dataset which was available on kaggle and was able to do the feature engineering. We had some progress which we have included in the code.
•	But converting the words into vectors required natural language processing which we are not aware of, so we went ahead and used the inbuilt dataset available by keras. 

•	The model has to train on over 25000 reviews, hence the model training was slow when working with classifiers like naive bayes etc. 
•	We have planned to try with recurrent neural networks too, but the algorithm was too complicated to construct and due to time constraint, we were unable to make any progress in it. Concerning data, team tried use reviews or sentiments collected from social media like twitter and facebook, but language modelling had to be done on the data collected. To perform the language modelling, we went ahead and learnt about natural language processing and sentimental analysis. But to analyse and work on the huge amount of data has taken lot of time.   
•	To understand the terminology used in tensorflow, keras and python was difficult as tensorflow and keras is updated regularly. 
•	Has not thrown any errors but had to deal with the warnings it generated.
•	We were able to eliminate most of the warning by understanding the terminology used.

7. Plots
Setting up the environment was time consuming.
Initially, we went ahead and used the imdb dataset which was available on kaggle and was able to do the feature engineering . We had some progress which we have included in the code.
Showing the first 5 rows of the imdb dataset downloaded from kaggle and modified. Using df.head(5) to show the first 5 rows. We could analyse features like director name, number of critics for reviews, duration of the film, gross, imdb score etc. 


We tried to plot the maximum imdb scores of each country. Following image shows that 9.3 was the maximum imdb score of a movie by the USA. We used plotly to plot this.

 ![image](https://user-images.githubusercontent.com/26388650/43681256-3f60e39c-9803-11e8-817b-fe1dd7331f6d.png)



 But converting the words into vectors required natural language processing which we are not aware of, so we went ahead and used the inbuilt dataset available by keras. 


8. Evaluation Matrix
	8.1 PLA - evaluation and efficiency:

	With a small amount of data as input to a simple model, we were able to achieve an accuracy rate of approximately 85.97%. If we make our model a bit more complex or if we provided more data for training our model, we can reach an even higher accuracy level.


 8.2 Convolutional neural network - evaluation and efficiency:

The convolutional neural network has generated better accuracy when compared to PLA. We were successful in evaluating the model with an accuracy of 88.15% . We have also calculated the loss and accuracy of each epoch for the purposes of back propagation.
 
9. Justification for the approaches used
To work on the huge amount of data, we needed a sophisticated algorithm which runs fast when compared to other algorithms. As our project is Sentiment Analysis where the input is classified as positive or negative, we need a binary classification algorithm that classifies our input into one of the two classes.
Hence we came down to analyse the efficiency of PLA and neural network and evaluate which algorithm worked well. 
10. References

1.	“Sentiment Analysis on Movie Reviews | Kaggle,” Sentiment Analysis on Movie Reviews | Kaggle. [Online]. Available: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews. [Accessed: 25-May-2017].

2.	“CS231n Convolutional Neural Networks for Visual Recognition,” CS231n Convolutional Neural Networks for Visual Recognition. [Online]. Available: http://cs231n.github.io/convolutional-networks/. [Accessed: 25-May-2017].

3.	A.Geitgey, “Machine Learning is Fun! Part 3: Deep Learning and Convolutional Neural Networks,” Medium, 13-Jun-2016. [Online]. Available: https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721. [Accessed: 25-May-2017].

4.	“Introduction to Python Deep Learning with Keras,” Machine Learning Mastery, 08-Oct-2016. [Online]. Available: http://machinelearningmastery.com/introduction-python-deep-learning-library-keras/. [Accessed: 25-May-2017].

5.	“Convolutional Neural Networks  |  TensorFlow,” TensorFlow. [Online]. Available: https://www.tensorflow.org/tutorials/deep_cnn. [Accessed: 25-May-2017]






