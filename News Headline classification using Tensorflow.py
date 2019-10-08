#!/usr/bin/env python
# coding: utf-8

# # News Aggregator
# ### model for news headline classification

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np 
import pandas as pd 
import tensorflow as tf
import string


# In[2]:


from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# ### Data set from kaggle
# ###### https://www.kaggle.com/uciml/news-aggregator-dataset

# In[4]:


data = pd.read_csv('uci-news-aggregator.csv', usecols=['TITLE', 'CATEGORY'])


# In[5]:


data.head()


# In[6]:


# A look at how many records we have per category
data.CATEGORY.value_counts()


# In[7]:


# taking the maximum chunk of every category
num_of_categories = 45000


# In[8]:


# suffling the data frame
shuffled = data.reindex(np.random.permutation(data.index))
# seperating each category equally
e = shuffled[shuffled['CATEGORY'] == 'e'][:num_of_categories]
b = shuffled[shuffled['CATEGORY'] == 'b'][:num_of_categories]
t = shuffled[shuffled['CATEGORY'] == 't'][:num_of_categories]
m = shuffled[shuffled['CATEGORY'] == 'm'][:num_of_categories]
# combining all the categories to a data frame
concated = pd.concat([e,b,t,m], ignore_index=True)
# Re-Shuffling the dataset
concated = concated.reindex(np.random.permutation(concated.index))
# making a LABEL column and filling with 0
concated['LABEL'] = 0


# In[9]:


concated.head()


# In[10]:


# numerical map of categories to label
concated.loc[concated['CATEGORY'] == 'e', 'LABEL'] = 0
concated.loc[concated['CATEGORY'] == 'b', 'LABEL'] = 1
concated.loc[concated['CATEGORY'] == 't', 'LABEL'] = 2
concated.loc[concated['CATEGORY'] == 'm', 'LABEL'] = 3
# making a label column in to a One Hot Encoding
labels = tf.keras.utils.to_categorical(concated['LABEL'], num_classes=4)


# In[11]:


concated.head()


# In[12]:


# a ruff estimate of the number of features
n_most_common_words = 8000
# max len for padding
max_len = 130
# creating a tokenizer
tokenizer = Tokenizer(num_words=n_most_common_words, filters=string.punctuation, lower=True)
# making the tokenizer train on our text i,e News Headlines
tokenizer.fit_on_texts(concated['TITLE'].values)
# Transforming out text into an integer sequence i.e list
sequences = tokenizer.texts_to_sequences(concated['TITLE'].values)
# Padding the lists to level the playfield
X = pad_sequences(sequences, maxlen=max_len)


# In[13]:


# Splitting considered data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)


# In[14]:


# no of Epochs for Model Training
epochs = 10
# Output Dimensions for Embedded Layer
emb_dim = 128
# Batch size for training
batch_size = 256


# In[16]:


# Creating a LSTM model
model = tf.keras.Sequential()
model.add(layers.Embedding(n_most_common_words, emb_dim, input_length=X_train.shape[1]))
model.add(layers.SpatialDropout1D(0.5, ))
model.add(layers.LSTM(64, dropout=0.7, recurrent_dropout=0.7, ))
model.add(layers.Dense(4, activation='softmax'))


# In[17]:


# compiling the layer 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[18]:


print(model.summary())


# In[19]:


# Training the model 
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])


# In[20]:


# model = tf.keras.models.load_model('News_Aggregator_LSTM_v1.h5')


# In[21]:


# Evaluating the model against the test data
accr = model.evaluate(X_test,y_test, verbose=0);
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[22]:


# Visuaizing how accuracy and loss progressed
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[24]:


# Evaluation
txt = ["Regular fast food eating linked to fertility issues in women"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len)
pred = model.predict(padded)
labels = ['entertainment', 'bussiness', 'science/tech', 'health']
print(pred, labels[np.argmax(pred)])

