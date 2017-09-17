
# coding: utf-8

# In[1]:

import nltk
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from IPython.display import Image
pd.options.display.max_colwidth = 100


# In[2]:

df = pd.read_table('sms.tsv',header = None,names = ['label','text'])
df.head()


# In[3]:

"""Examine the class distribution"""
print(df.label.value_counts())
print("Total:",df.label.value_counts().sum())


# In[4]:

"""Convert spam to 1 (a success) and ham to a 0 (failure)"""
df['label_num'] = df.label.map({'ham':0, 'spam':1})
df.head()


# In[5]:

"""Define the features and output"""
X = df['text'] # The features that we will be using -> our X vector
y = df['label_num'] # The output class for our features
print(X.shape)
print(y.shape)


# In[6]:

"""Split the data into the training and testing sets"""
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape) # Feature matrix of the training set
print(X_test.shape) # Feature matrix of the testing set
print(y_train.shape) # Class vector of the training set
print(y_test.shape) # Class vector of the testing set


# In[7]:

"""Instantiate the vectorizer"""
ngram = 2
default_stopwords = set(nltk.corpus.stopwords.words('english')) # Create the default stopwords set
vect = CountVectorizer(ngram_range=(ngram,ngram),stop_words = default_stopwords)


# In[8]:

"""Learn training data vocabulary, then use it to create a document-term matrix"""
get_ipython().magic('time X_train_dtm = vect.fit_transform(X_train)')


# In[9]:

X_train_dtm


# # What is a Sparse Matrix?

# In[10]:

dm = pd.DataFrame(X_train_dtm.toarray(),columns = vect.get_feature_names())
# dm.iloc[dm[0].nonzero()[0]]
dm.iloc[:,1000:]


# # Have to be careful with this!

# In[11]:

Image('pics/memory usage.png')


# In[12]:

"""Transform testing data (using fitted vocabulary) into a document-term matrix"""
X_test_dtm = vect.transform(X_test)
X_test_dtm


# In[13]:

"""Instantiate our Logistic Regression Model with default parameters"""
logreg = LogisticRegression()


# In[14]:

"""Fit our training feature matrix and output vector"""
get_ipython().magic('time logreg.fit(X_train_dtm,y_train)')


# # Predict

# In[15]:

y_pred_class_logreg = logreg.predict(X_test_dtm)


# In[16]:

print(np.round(metrics.accuracy_score(y_test, y_pred_class_logreg) * 100,2),'% accurate')


# In[17]:

"""Custom function to show the features and their parameters"""
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

show_most_informative_features(vect,logreg)


# # Metrics

# In[18]:

"""Confusion matrix"""
Image('pics/confusion.png')


# In[19]:

print(metrics.confusion_matrix(y_test, y_pred_class_logreg)) # The actual confusion matrix


# In[20]:

confusion = metrics.confusion_matrix(y_test, y_pred_class_logreg)
TP = confusion[1, 1] # True positive
TN = confusion[0, 0] # True Negative
FP = confusion[0, 1] # False positive
FN = confusion[1, 0] # False Negative
print('True Positive:',TP)
print('True Negative:',TN)
print('False Positive:',FP)
print('False Negative:',FN)


# In[21]:

"""Determining the class is done by the probability cut off at 0.5"""
[i for i in zip(logreg.predict(X_test_dtm),logreg.predict_proba(X_test_dtm))]


# In[22]:

# print message text for the false positives (ham incorrectly classified as spam)
X_test[y_test < y_pred_class_logreg]


# In[23]:

# print message text for the false positives (ham incorrectly classified as spam)
X_test[y_test > y_pred_class_logreg][:10]

