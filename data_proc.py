import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from string import ascii_lowercase

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
scaler = StandardScaler()

# Data extraction
emails_data = pd.read_csv('./training_data/emails.csv')
df = pd.DataFrame(data=emails_data)
df = df.dropna()

letters = list(ascii_lowercase)
df = df.drop(columns=letters, errors='ignore')
num_words = df.drop(columns=['Email No.', 'Prediction'])
num_words['num_of_words'] = num_words.sum(axis=1)

# Ham Mail vs Spam Mail visualisation
#plt.figure(figsize=(12,6))
#plt.pie(df['Prediction'].value_counts(), labels=['Ham mails', 'Spam mails'], autopct='%1.1f%%')
#plt.show()

# Data preprocessing and engineering
total_words_count = num_words.mean(axis=0)
bag_words = df.drop(columns=['Email No.', 'Prediction'])

fetch_features = bag_words.columns

word_counts_ratio = df.drop(columns=['Email No.', 'Prediction'])
word_counts_ratio /= total_words_count
word_counts_ratio['sum_ratios'] = word_counts_ratio.sum(axis=1)

corr = word_counts_ratio['sum_ratios'].corr(df['Prediction'].astype(int))
#print(corr)

#repeated words
repeated_words = (bag_words > 2).astype(int)
repeated_words['rep_words'] = repeated_words.sum(axis=1)

corr = repeated_words['rep_words'].corr(df['Prediction'].astype(int))
#print(corr)

#most used words of spam mails
spam_mail_words = []

chi2_selector = SelectKBest(chi2, k = 1000) #k value can be altered
X_kbest = chi2_selector.fit_transform(bag_words, df['Prediction'])
top_spam_words = bag_words.columns[chi2_selector.get_support()]

corrs = bag_words.corrwith(df['Prediction'])
corrs = (corrs >= 0.1).astype(int)
corrs = corrs[corrs != 0]

spam_words = [f for f in top_spam_words if f in corrs]
spam_words_cols = df[spam_words]

train_x = pd.DataFrame({
    'word_ratios': word_counts_ratio['sum_ratios'],
    'repeated_words': repeated_words['rep_words']
})

for i in range(len(spam_words)):
    idx = spam_words[i]
    pd_series_col = pd.Series(spam_words_cols[idx])
    train_x[idx] = pd_series_col

#to reduce data skewness
for col in train_x.columns:
    train_x[col] = np.log1p(train_x[col])


train_x = scaler.fit_transform(train_x)
train_x = np.array(train_x)
train_y = np.array(df['Prediction'])
