# Spam_mail_detection

This is a detection system consisting of two parts: 
  - A logistic regression model built from scratch
  - A neural network model built using tensorflow

The dataset has over 5172 mails and 1000+ features comprising of different words commonly present in most mails.

Both the models are designed to perform spam mail detection using a spam mail identifier dataset.

Model performance was evaluated using accuracy, F1 score, and confusion matrices.

A pipeline is designed for users to test the different models using the mail.txt file.

The dataset: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data

# Model explanation

## Logistic regression model

  - The model is built from scratch using modified formulas derived from linear regression and gradient descent
  - It makes use of L2/ridge regularization method to avoid overfitting
  - The accuracy of the model is ~88.6%

## Neural network model
  - The model consists of 4 Dense Layers
  - It makes use of extra features such as Regularization, Early Stopping and Adam optimizer for improving accuracy
  - The accuracy of the model is ~95.5%

# Dependencies
  - NumPy
  - Pandas
  - TensorFlow
  - Scikit-learn
  - Matplotlib

# Usage

  - It is recommended that all the dependencies are downloaded for the project to be used properly
  - The logistic regression model may be time consuming since it does not use NumPy vectorization
  - Several commented print statements are provided to observe the performance of the two models
  - Content of mail.txt model can be modified to test the two models

# Author

  - Shriyans Sharma
