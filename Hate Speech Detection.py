#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[20]:


import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopword=set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")


# In[21]:


data=pd.read_csv("twitter.csv")
print(data.head())


# In[23]:


data["labels"]=data["class"].map({0:"Hate Speech", 1:"Offensive Speech", 2:"No Hate and Offensive speech"})


# In[24]:


data=data[["tweet","labels"]]


# In[25]:


data.head()


# In[38]:


def clean (text):
    text = str (text).lower()
    text = re. sub('[.?]', '', text)
    text = re. sub('https?://\S+|www.\S+', '', text)
    text = re. sub('<.?>+', '', text)
    text = re. sub('[%s]' % re. escape(string. punctuation), '', text)
    text = re. sub('\n', '', text)
    text = re. sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ". join(text)
    text = [stemmer. stem(word) for word in text.split(' ')]
    text=" ". join(text)
    return text
data["tweet"] = data["tweet"]. apply(clean)
print(data.head())


# In[40]:


x=np.array(data["tweet"])
y=np.array(data["labels"])
cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= 0.33, random_state= 42)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[45]:


y_pred=model.predict(X_test)


# In[46]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[47]:


i="I will kill you"
i = cv.transform([i]).toarray()
print(model.predict(i))


# In[48]:


i="you are too bad and I dont like your attitude"
i = cv.transform([i]).toarray()
print(model.predict(i))


# In[ ]:





# In[ ]:




