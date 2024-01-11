#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('C:/FAULT_DIAG_PROJ/CWRU_dataset/48k_drive_end/0hp/0hp_all_faults.csv')


# In[3]:


# Data preprocessing
win_len = 784
stride = 300
X = []
Y = []


# In[4]:


for k in df['fault'].unique():
    df_temp_2 = df[df['fault'] == k]

    for i in np.arange(0, len(df_temp_2) - (win_len), stride):
        temp = df_temp_2.iloc[i:i + win_len, :-1].values
        temp = temp.reshape((1, -1))
        X.append(temp)
        Y.append(df_temp_2.iloc[i + win_len, -1])


# In[5]:


X = np.array(X)
X = X.reshape((X.shape[0], 28, 28, 1))
Y = np.array(Y)


# In[6]:


# One-hot encode the target variable
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
OHE_Y = to_categorical(encoded_Y)


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, OHE_Y, test_size=0.3, shuffle=True)


# In[8]:


# Create the CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='tanh', input_shape=(X.shape[1], X.shape[2], 1), padding='same'))
cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
cnn_model.add(Conv2D(64, (3, 3), activation='tanh', padding='same'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='tanh'))
cnn_model.add(Dense(len(df['fault'].unique()), activation='softmax'))


# In[9]:


# Change Activation Functions
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='tanh', input_shape=(X.shape[1], X.shape[2], 1), padding='same'))
cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
cnn_model.add(Conv2D(64, (3, 3), activation='tanh', padding='same'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='tanh'))
cnn_model.add(Dense(len(df['fault'].unique()), activation='softmax'))


# In[10]:


# Compile the model
cnn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[11]:


# Set the number of epochs to 50
epochs = 50


# In[12]:


# Train the CNN model
history = cnn_model.fit(X_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(X_test, y_test), shuffle=True)


# In[13]:


# Function to inverse transform predictions
def inv_Transform_result(y_pred):
    y_pred = y_pred.argmax(axis=1)
    y_pred = encoder.inverse_transform(y_pred)
    return y_pred


# In[14]:


# Predictions on the test set
y_pred = cnn_model.predict(X_test)
Y_pred = inv_Transform_result(y_pred)
Y_test = inv_Transform_result(y_test)


# In[15]:


# Confusion Matrix
plt.figure(figsize=(10, 10))
cm = confusion_matrix(Y_test, Y_pred, normalize='true')
f = sns.heatmap(cm, annot=True, xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.show()


# In[16]:


# Classification Report
print("Classification Report:")
print(classification_report(Y_test, Y_pred, target_names=encoder.classes_))


# In[17]:


# Additional Performance Metrics
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)


# In[18]:


print("\nAdditional Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print("Precision per class:")
for fault, prec in zip(encoder.classes_, precision):
    print(f"{fault}: {prec:.4f}")
print("Recall per class:")
for fault, rec in zip(encoder.classes_, recall):
    print(f"{fault}: {rec:.4f}")
print("F1 Score per class:")
for fault, f1 in zip(encoder.classes_, f1_score):
    print(f"{fault}: {f1:.4f}")


# In[19]:


# Visualize Results
num_samples_to_visualize = 5


# In[20]:


# Randomly select some samples from the test set
random_indices = np.random.choice(len(X_test), num_samples_to_visualize, replace=False)
sample_images = X_test[random_indices]
true_labels = Y_test[random_indices]


# In[21]:


# Predict the labels for the selected samples
predicted_labels = inv_Transform_result(cnn_model.predict(sample_images))


# In[22]:


# Plot the selected samples along with true and predicted labels
plt.figure(figsize=(15, 8))
for i in range(num_samples_to_visualize):
    plt.subplot(1, num_samples_to_visualize, i + 1)
    plt.imshow(sample_images[i, :, :, 0], cmap='gray')
    plt.title(f'True: {true_labels[i]}\nPredicted: {predicted_labels[i]}')
    plt.axis('off')

plt.show()


# In[ ]:




