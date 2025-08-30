# Spam_mail_detection

This is a detection system consisting of two parts: 
  - A logistic regression model built from scratch
  - A neural network model built using tensorflow

The dataset has over 5172 mails and 3000 features comprising of common words in most mails.

Both the models are designed to perform spam mail detection using a spam mail identifier dataset.

Model performance is evaluated using accuracy, F1 score, and confusion matrices.

A pipeline is designed for users to test the different models using the mail.txt file.

The dataset: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data


# Model explanation

## Logistic regression model

  - The model is built from scratch using modified formulas derived from linear regression, and gradient descent
  - It uses L2/ridge regularization method to avoid overfitting
  - The accuracy of the model is ~93.3%

  <img width="550" height="292" alt="image" src="https://github.com/user-attachments/assets/ef91e68f-b7df-4a76-94ab-1ceaa25701d5" />


  <img width="550" height="292" alt="image" src="https://github.com/user-attachments/assets/dbcb7aef-e16e-431c-a907-89b386fa75ef" />


## Neural network model
  - The model consists of 4 Dense Layers
  - It uses features such as Regularization, Early Stopping and Adam optimizer for improving accuracy
  - The accuracy of the model is ~96.4%

  <img width="550" height="292" alt="image" src="https://github.com/user-attachments/assets/ca0b4c28-3c08-45b6-9b2b-7f4648e03d72" />
  

  <img width="550" height="292" alt="image" src="https://github.com/user-attachments/assets/1c646355-1e6e-4de9-9805-974636e5de04" />


# Files
  - emails.csv: dataset file
  - mail.txt: user can modify contents to test the two models
  - data_proc.py: data processing of the dataset
  - spam_filter_lr.py: logistic regression model
  - spam_filter_nn.py: neural network model
  - predict_mail.py: to test the two models using mail.txt

# Dependencies
  - NumPy
  - Pandas
  - TensorFlow
  - Scikit-learn
  - Matplotlib


# Usage

  - Run pip install -r requirements.txt to install all dependencies required to run the code
  - The logistic regression model may be time consuming since it does not use NumPy vectorization
  - Several commented print statements are provided to observe the performance of the two models
  - Content of mail.txt can be modified to test the two models
  - The user is required to only use the predict_mail.py and mail.txt file to test both models
  - K value of the chi2_selector in data_proc.py may be changed:
    - Increasing would increase training time but improve performance
    - Decreasing would decrease performance but improve training time
   
# Limitations

  This model was trained on a specific dataset, which does not include all words commonly found in spam emails. 
  Consequently, predictions may be inaccurate for emails containing vocabulary not represented in the training data.

# Author

  - Shriyans Sharma
