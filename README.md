# CNN Dog-Cat Classification Model
This repository contains a Convolutional Neural Network (CNN) model for classifying images of dogs and cats. The model is created from scratch using Python and popular deep learning libraries such as TensorFlow and Keras. Additionally, a pipeline has been deployed for prediction, which includes an HTML user interface for easy interaction.

## Model Overview
The CNN model is designed to classify images into two categories: "dog" and "cat". It consists of the following layers:

Input layer: The images are fed as input to the model.
Convolutional layers: These layers extract features from the input images using filters.
Pooling layers: Pooling operations downsample the features and reduce spatial dimensions.
Fully connected layers: These layers connect all neurons from the previous layer to the subsequent layer.
Output layer: The final layer produces the classification probabilities for the "dog" and "cat" classes.
## Model Parameters
The CNN model is trained with the following parameters:

Number of convolutional layers: [Specify the number of convolutional layers]
Number of filters per convolutional layer: [Specify the number of filters for each convolutional layer]
Pooling type: [Specify the type of pooling used]
Number of fully connected layers: [Specify the number of fully connected layers]
Activation functions: [Specify the activation functions used in the model]
Optimizer: [Specify the optimizer used during training]
Learning rate: [Specify the learning rate used during training]
Batch size: [Specify the batch size used during training]
Number of epochs: [Specify the number of training epochs]
Deployment Pipeline
To deploy the model and create a prediction pipeline, the following steps were performed:

## Data preprocessing: 
The dataset containing images of dogs and cats was preprocessed to ensure consistency and proper formatting.

Model training: The CNN model was trained on the preprocessed dataset using the specified parameters.
Model evaluation: The trained model was evaluated using various performance metrics such as accuracy, precision, and recall.
Pipeline creation: An HTML user interface was developed to allow users to upload images and obtain predictions from the trained model.
Model deployment: The pipeline, along with the trained model, was deployed on a server or hosting platform for accessibility.
Usage
To use the model and prediction pipeline, follow these steps:

## Clone or download this repository to your local machine.
Install the required dependencies listed in the requirements.txt file.
Run the predict.py script to start the prediction server.
Open the HTML user interface by accessing the provided URL or navigating to index.html in your web browser.
Upload an image of a dog or cat using the user interface.
The trained model will predict the class of the uploaded image, displaying the result on the user interface.
