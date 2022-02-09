# Dog-Breed-Identification---ML-project-

In this Machine Learning project, I developed a model, which would recognize dog breeds. Below, I provide more detailed information about it.


# Objective

Several times I saw a dog and didn't know the breed of it. Also I wanted to go deeper in Machine Learning, CNN and image recognition, so I decided to take the advantage and start with a simple image identification project, that I could develop my skills and also to have an app, which would tell the breeds.


# Dataset

For this project, I used the dataset from the Kaggle competition: Dog Breed Identification. The dataset consist of 2 subsets: train and test. Train subset also includes the "labels" as a separate part of it. Both subsets have more than 10000 images for training purposes.


# Model

To build neural network model I used InceptionV3 - a Keras image classification pre-trained model. This model had already been trained using a dataset of 1,000 classes from the original ImageNet dataset which was trained with over 1 million training images*. I set the parameters in accordance of Keras API references recommendation and fine tune them. After fitting the model, evaluated the Accuracy and Loss of the prediction.


# Findings

The accuracy of the model reaches 90%, which means that with fine tuning the parameters, the model can be developped further. This model is also deployed in an Streamlit app: https://share.streamlit.io/akbarhuseynov23/dog-breed-recognition---ml-final-project/main/dog_breeds.py

Source: https://www.intel.com/content/www/us/en/develop/articles/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic.html
