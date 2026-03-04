import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from data_proc import train_x, train_y

X_train, X_test, Y_train, Y_test = train_test_split(
    train_x, train_y,
    test_size=0.30,
    random_state=42,
    stratify=train_y 
)

# The 4-Layered Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(408,), dtype=np.float64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, kernel_regularizer = tf.keras.regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation='relu'), 
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

#to stop training early to avoid overfitting
earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), callbacks=[earlyStopping])

# Model performance markers
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int).flatten()

cfm = confusion_matrix(Y_test, y_pred)
#print("Neural network model confusion matrix: \n", cfm)

acc = accuracy_score(Y_test, y_pred)
#print("Neural network accuracy score: ", acc)
f1 = f1_score(Y_test, y_pred, average='weighted')
#print("Neural network f1 score: ", f1)

# Visualisation of model performance

#plotting graph of loss of model
plt.figure(figsize=(12,6))
plt.title("Neural network model loss")
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.show()

#plotting graph of accuracy of model
plt.figure(figsize=(12,6))
plt.title("Neural network model accuracy")
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.show()

#plotting graphs of loss and val_loss of model
plt.figure(figsize=(12,6))
plt.title("Neural network model loss and val_loss")
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='magenta', label='val_loss')
plt.legend()
plt.show()

#plotting graphs of accuracy and val_accuracy of model
plt.figure(figsize=(12,6))
plt.title("Neural network model accuracy and val_accuracy")
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='magenta', label='val_accuracy')
plt.legend()
plt.show()