#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction

# ### Import Libraries

# In[548]:


import pandas as pd #Read file
import numpy as np #Matrix multiplication
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[549]:


df = pd.read_csv(r"C:\Users\Khadiza\Downloads\Heart Attack Data Set.csv")


# In[550]:


df #All Dataset Row


# In[551]:


df.head() #Top 5 row


# In[552]:


df.head(20)


# In[553]:


df.tail() #Bottom 5 Row


# ### Info of Dataset

# In[554]:


df.shape #To see how many rows and column


# In[555]:


#0 = no disease, 1 = disease
df['target'].value_counts() 


# In[556]:


df['sex'].value_counts()  # # 1 = male; 0 = female  //Numerical value , unique value


# In[557]:


#How to see all columns 
df.columns


# In[558]:


#Rename columns name 
df = df.rename(columns = {"sex":"gender"})


# In[559]:


df.head()


# In[560]:


df.describe() #Data Minimum And Max Value


# In[561]:


#The correlation matrix is a square matrix that shows how strongly pairs of variables are related to each other.
df.corr()


# In[562]:


df.info()


# In[563]:


df.isnull().sum()  #Checking null value


# In[564]:


#Or other function to check null value
df.isna().sum()


# In[565]:


df.isnull().values.any() #IF there any null value reture True otherwise false


# In[566]:


#Delete null value from row 
#df = df.dropna()   then df.head()


# In[567]:


# Multiple columns remove= df.drop(columns = ["slope","ca"],axis = 1) 
# df.head()

#Single Column Delete
# df = df.drop(['thal'],axis = 1) 

#axis = 1 means delete column null value, axis = 0 means delete column from row  then df.head()


# ### Data Visualization

# In[568]:


#sn.countplot(df["target"])  
#for visualize


# In[569]:


#gender (1 = male; 0 = female) 
# 0 = no disease, 1 = disease

sn.countplot(x='gender' ,hue='target' ,data=df,palette='colorblind' ,edgecolor=sn.color_palette('dark',n_colors=1))


# In[570]:


# normal attribute and class attribute feature alada korar jonno
x_input = df.iloc[:,:-1]


# In[571]:


x_input


# In[572]:


y_output = df.iloc[:,13]


# In[573]:


y_output


# ### Splitting Data

# In[587]:


# One will use for train and one will use for testing

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state= 137)  #20% for testing


# In[588]:


len(x_train), len(x_test)


# # Random Forest Classifier 

# In[589]:


rf_clf = RandomForestClassifier(random_state=137) # For reproducibility
rf_clf.fit(x_train,y_train)


# In[590]:


rf_pred = rf_clf.predict(x_test) # Make predictions on the test data


# In[591]:


print("Random Forest MSE:",mean_squared_error(y_test,rf_pred))
print("Random Forest Accuracy:",accuracy_score(y_test,rf_pred))
print("Random Forest Precision:",precision_score(y_test,rf_pred))
print("Random Forest Recall:",recall_score(y_test,rf_pred))
print("Random Forest F1 Score:",f1_score(y_test,rf_pred))
print("Random Forest Confusion Matrix:",confusion_matrix(y_test,rf_pred))


# In[592]:


# Another Way to find Accuuracy
rf_clf.score(x_test, y_test)


# # Decision Tree Classifier

# In[594]:


dc_clf = DecisionTreeClassifier(random_state=137)
dc_clf.fit(x_train,y_train) 


# In[595]:


dc_pred = dc_clf.predict(x_test) # Make predictions on the test data


# In[596]:


print("Decision Tree MSE:",mean_squared_error(y_test,dc_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test,dc_pred))
print("Decision Tree Precision:",precision_score(y_test,dc_pred))
print("Decision Tree Recall:",recall_score(y_test,dc_pred))
print("Decision Tree F1 Score:",f1_score(y_test,dc_pred))
print("Decision Tree Confusion Matrix:", confusion_matrix(y_test,dc_pred))


# In[597]:


# Another Way to find Accuuracy
dc_clf.score(x_test, y_test)


# # Naive Bayes

# In[598]:


nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)


# In[599]:


nb_pred = nb_clf.predict(x_test)


# In[600]:


print("Naive Bayes MSE:",mean_squared_error(y_test,nb_pred))
print("Naive Bayes Accuracy:",accuracy_score(y_test,nb_pred))
print("Naive Bayes Precision:",precision_score(y_test,nb_pred))
print("Naive Bayes Recall:",recall_score(y_test,dc_pred))
print("Naive Bayes F1 Score:",f1_score(y_test,dc_pred))
print("Naive Bayes Confusion Matrix:",confusion_matrix(y_test,nb_pred))


# In[ ]:





# In[ ]:




