This Project is part of TensorFlow Developer Spealization Conducted by DeepLearning AI and is the final project of course of of this spealization, Convolutional Neural Network in TensorFlow

The Project aims to create a CNN model to classify Alphabetical Sign Language Images, trained using MNIST Sign Languages Dataset, which is provided on https://www.kaggle.com/datasets/datamunge/sign-language-mnist

The MNIST dataset containst of 27,455 images for training and 7172 iamges for testing. This dataset is represented on csv format, so that converting to array, one cantaining label and one containing features is necessary. 

ImageDataGerators are used to generate the training and validation generator for training model. The model containst of 2 Conv2D layers with ReLU activation function, 1 Dropout layer with rate 0.2, and 2 Dense layer, one having 512 nodes, and one having 26 nodes and acting as output of the network 
 