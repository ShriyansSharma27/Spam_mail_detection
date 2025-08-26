import re
import pandas as pd
from data_proc import *
from spam_filter_lr import *
from spam_filter_nn import * 

file_path="mail.txt" #contents of the text file can be altered to test 
cols = df.columns
words = []
count_words = {}

#extracting words only from the mail txt file
try:
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = re.sub(r'[^a-zA-Z\s]', '', line)
            line_split = line.strip().split(' ')
            for wrd in line_split:
                words.append(wrd.lower())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

#to store count of each word
for i in range(len(cols)):
    count_words[cols[i]] = [words.count(cols[i])]

input_mail_df = pd.DataFrame(count_words)
input_mail_df = input_mail_df.drop(columns=['Email No.', 'Prediction'])

#ratio (no. of words in mail/ avg words in each mail)
wrd_ratios = [((input_mail_df / total_words_count).sum(axis=1).sum())]

#num of repeated words in mail
mail_rep_words = (bag_words > 2).astype(int)
rep_words = [(mail_rep_words.sum(axis=1).sum())]

input_mail_features = pd.DataFrame({
    'word_ratios': wrd_ratios,
    'repeated_words': rep_words
})

for col in spam_words:
    if col in words:
        input_mail_features[col] = [(input_mail_df[col].sum())]
    else:
        input_mail_features[col] = [0]

#to reduce data skewness
for i in input_mail_features:
    input_mail_features[i] = np.log1p(input_mail_features[i])

input_mail_features = scaler.transform(input_mail_features)
input_mail_features = input_mail_features[0]


z = np.dot(weights, input_mail_features) + b

pred_lr = 1 / (1 + np.exp(-z))
pred_lr = ((pred_lr>=0.5)).astype(int)


print("Logistic regression model prediction: ", pred_lr)


pred_nn = model.predict(np.array(input_mail_features).reshape(1,-1))
pred_nn = pred_nn[0][0]
pred_nn = ((pred_nn>=0.5)).astype(int)

print("Neural network model prediction: ", pred_nn)


