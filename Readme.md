# Sign Language Classification

## Overview

This project is part of the TensorFlow Developer Specialization conducted by DeepLearning.AI and serves as the final project for the "Convolutional Neural Networks in TensorFlow" course within this specialization.

The objective of this project is to develop a Convolutional Neural Network (CNN) model that can classify images of alphabetical sign language. The model is trained using the MNIST Sign Language Dataset, splitted into training and testing dataset. To download the full dataset, you can go to the original dataset link on kaggle, which can be accessed [here](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

## Dataset

The MNIST Sign Language Dataset consists of images representing the 26 letters of the alphabet in American Sign Language (ASL).

- **Training Images**: 27,455
- **Testing Images**: 7,172
- **Format**: The dataset is provided in CSV format, so preprocessing involves converting the data into arrays: one for labels and one for features.

## Model Architecture

The model is built using TensorFlow and Keras and includes the following layers:

1. **Input Layer**: Receives the preprocessed images.
2. **Convolutional Layers**: 
   - 2 Conv2D layers with ReLU activation function.
3. **Dropout Layer**: A dropout layer with a rate of 0.2 to prevent overfitting.
4. **Fully Connected Layers**: 
   - 1 Dense layer with 512 nodes.
   - 1 Output Dense layer with 26 nodes, representing the 26 letters of the alphabet, using the Softmax activation function.

The architecture is designed to effectively learn and classify the features of hand gestures representing different letters.

## Training

- **Data Augmentation**: ImageDataGenerators are used to generate the training and validation sets for training the model.
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 30

The model is trained to accurately classify each image into one of the 26 classes corresponding to the ASL alphabet.

## Future Work

- **Real-Time Detection**: Implement a real-time sign language detection system using a webcam.
- **Gesture Sequence Recognition**: Develop a system to recognize sequences of gestures to form words and sentences.
- **Enhanced Data Augmentation**: Apply more complex data augmentation techniques to improve model robustness.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature suggestions.

