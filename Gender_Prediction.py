#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.compat import lzip


# In[2]:


# loading dataset
df = pd.read_csv('name_gender.csv',names=['Names', 'Gender', 'irrelavant'])

#checking top 5 observations of the dataset
df.head()


# In[3]:


# checking shape of the dataset
df.shape


# In[4]:


# checking missing values of the datset
df.isnull().sum()


# In[5]:


# checking value counts of the target variable
df.Gender.value_counts()


# # With NLP

# In[6]:


# creating a function for fetching last character of the string
def features(word):
    word = str(word).lower()
    return {'last_letter': word[-1:]}


# In[7]:


# Applying upper created function on a word
print(features('Sam'))


# In[8]:


# importing random library to shuffle
import random

# importing nltk library 
import nltk


# In[9]:


# creating a list by list comprehension function
# fetching name and gender from dataset using lzip function
names = [(i, j) for i, j in lzip(df['Names'], df['Gender'])]


# In[10]:


# printing names
print(names)


# In[11]:


# run mult times v get shuffled names to reduce bias
random.shuffle(names)


# In[12]:


# fetching 20 observations from names
for name,gender in names[:20]:
    print('Name: ', name, 'Gender:',gender)


# In[13]:


# make a feature set for all the names
feature_sets = [(features(name.lower()), gender) for name, gender in names]


# In[14]:


# length of the feature_sets
len(feature_sets)


# In[15]:


# fetching 20 observations from feature_sets
for dict,gender in feature_sets[:20]:
    print(dict,gender)


# In[16]:


# printing length of the feature_sets
print(len(feature_sets))


# In[17]:


# Making a testing data set and training data set

train_set = feature_sets[30000:]

test_set = feature_sets[:30000]


# In[18]:


# training the data
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[19]:


# Now we test it against names
print(classifier.classify(features('Samy')))


# In[20]:


# Now we test it against names
print(classifier.classify(features('Sam')))


# In[21]:


# Testing the accuracy of our classifier

print(nltk.classify.accuracy(classifier, test_set)*100)


# # With ML

# In[22]:


# creating new function to fetch 1st character
# 1st two characters
# 1st three characters
# similarly last character
# last 2 characters
# last 3 characters
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }
# printing function with a name John 
print(features("Pratik"))


# In[23]:


# assinging train split as 0.8
TRAIN_SPLIT = 0.8


# In[24]:


# Vectorize the features function
features = np.vectorize(features)
print(features(["Anna", "Hannah", "Paul"]))
# [ array({'first2-letters': 'an', 'last-letter': 'a', 'first-letter': 'a', 'last2-letters': 'na', 'last3-letters': 'nna', 'first3-letters': 'ann'}, dtype=object)
#   array({'first2-letters': 'ha', 'last-letter': 'h', 'first-letter': 'h', 'last2-letters': 'ah', 'last3-letters': 'nah', 'first3-letters': 'han'}, dtype=object)
#   array({'first2-letters': 'pa', 'last-letter': 'l', 'first-letter': 'p', 'last2-letters': 'ul', 'last3-letters': 'aul', 'first3-letters': 'pau'}, dtype=object)]
 
# Extract the features for the whole dataset
X = features(df['Names']) # X contains the features
 
# Get the gender column
y = df['Gender']           # y contains the targets
 
# Test if we built the dataset correctly
print("Name: %s, features=%s, gender=%s" % (df[df['Names'].index==0], X[0], y[0]))
# Name: Mary, features={'first2-letters': 'ma', 'last-letter': 'y', 'first-letter': 'm', 'last2-letters': 'ry', 'last3-letters': 'ary', 'first3-letters': 'mar'}, gender=F


# In[25]:


# creating new dataframe as storing the previous dataset in it
df_new = df


# In[26]:


# Replacing All F and M with 0 and 1 respectively
df_new['Gender'].replace({'F':0,'M':1},inplace=True)


# In[27]:


# splitting data into train and test as in 80-20
from sklearn.utils import shuffle
X, y = shuffle(X, y)
X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
y_train, y_test = y[:int(TRAIN_SPLIT * len(y))], y[int(TRAIN_SPLIT * len(y)):]
 
# Check to see if the datasets add up
print(len(X_train), len(X_test), len(y_train), len(y_test))   # 76020 19005 76020 19005


# In[28]:


# applying DictVectorizer on the dataset
from sklearn.feature_extraction import DictVectorizer
 
var = features(["Mia", "Peter"])
vectorizer = DictVectorizer()
vectorizer.fit(var)
 
transformed = vectorizer.transform(var)
print(transformed)

print(type(transformed)) # <class 'scipy.sparse.csr.csr_matrix'>


# In[29]:


# transforming data with DictVectorizer
vectorizer = DictVectorizer()
X_train1 = vectorizer.fit_transform(X_train)


# # Decision tree

# In[30]:


# importing the DecisionTreeClassifier Algorithm
from sklearn.tree import DecisionTreeClassifier

# fitting data into model
DT = DecisionTreeClassifier().fit(X_train1, y_train)


# In[31]:


# checking the score as accuracy of the training data
print(DT.score(vectorizer.fit_transform(X_train), y_train))


# In[32]:


# checking the score as accuracy on testing data
print(DT.score(vectorizer.transform(X_test), y_test))


# # XGBoost

# In[33]:


# importing the XGBoostClassifier Algorithm
from xgboost import XGBClassifier

# fitting data into model
xg = XGBClassifier().fit(X_train1, y_train)


# In[34]:


# checking the score as accuracy of the training data
print(xg.score(vectorizer.fit_transform(X_train), y_train))


# In[35]:


# checking the score as accuracy on testing data
print(xg.score(vectorizer.transform(X_test), y_test))


# In[ ]:




