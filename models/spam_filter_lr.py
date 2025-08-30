import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from string import ascii_lowercase
from src.data_proc import train_x, train_y

# Visualisation 
cost_function = []
accuracy_scores = []
f1_scores = []

# Logistic regression model
epochs = 80
alpha = 0.1
lambda_val = 0.7 #for l2 regularization 
sample_size = len(train_x)
weights = [0] * len(train_x[0])
b = 0

for e in range(epochs):
    #print("Epoch: ", e)

    dj_db = 0
    dj_dw = [0] * len(weights)
    j_wb = 0 #cost

    preds = []
    actuals = []

    for i in range(sample_size):
        # z = w_i + b
        z = 0
        for j in range(len(weights)):
            z += (weights[j] * train_x[i][j])
        z += b

        f_wb_i = 1 / (1 + np.exp(-z)) # f_wb function
        loss_i = - (train_y[i] * math.log(f_wb_i)) - ( (1 - train_y[i]) * math.log(1 - f_wb_i))  #loss function 
        j_wb += loss_i

        dw_db = f_wb_i - train_y[i]
        dj_db += dw_db
        for j in range(len(dj_dw)):
            dj_dw[j] += (dw_db * train_x[i][j])
        
        preds.append((f_wb_i >= 0.4).astype(int))
        actuals.append(train_y[i])

    j_wb /= sample_size
    #print("Cost: ", j_wb)

    cost_function.append(j_wb)

    dj_db /= sample_size
    for i in range(len(dj_dw)):
        dj_dw[i] += (lambda_val * weights[i])
        dj_dw[i] /= sample_size

    # Update the parameters
    b -= (alpha * dj_db)
    for i in range(len(weights)):
        weights[i] -= (alpha * dj_dw[i])    

    acc_score = accuracy_score(actuals, preds)
    accuracy_scores.append(acc_score)
    #print("Accuracy score: ", acc_score)

    f1_sc = f1_score(actuals, preds, average='weighted')
    f1_scores.append(f1_sc)
    #print("F1 score: ", f1_sc)

    #if(e == (epochs - 1)):
        #print(confusion_matrix(actuals, preds))

# Visualisation of model performance
plt.figure(figsize=(12,6))
plt.title("Logistic regression model loss")
plt.plot([f for f in range(len(cost_function))], cost_function, label='Training loss', color='teal')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Logistic regression model accuracy")
plt.plot([f for f in range(len(cost_function))], accuracy_scores, label='Training accuracy', color='magenta')
plt.show()

