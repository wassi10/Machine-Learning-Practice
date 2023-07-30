#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction

# ### Import Libraries

# In[98]:


import pandas as pd #Read file
import numpy as np #Matrix multiplication
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[99]:


df = pd.read_csv(r"C:\Users\Khadiza\Downloads\Heart Attack Data Set.csv")


# In[100]:


df #All Dataset Row


# In[101]:


df.head() #Top 5 row


# In[102]:


df.head(20)


# In[103]:


df.tail() #Bottom 5 Row


# ### Info of Dataset

# In[104]:


df.shape #To see how many rows and column


# In[105]:


#0 = no disease, 1 = disease
df['target'].value_counts() 


# In[106]:


df['sex'].value_counts()  # # 1 = male; 0 = female  //Numerical value , unique value


# In[107]:


#How to see all columns 
df.columns


# In[108]:


#Rename columns name 
df = df.rename(columns = {"sex":"gender"})


# In[109]:


df.head()


# In[110]:


df.columns


# In[111]:


df.describe() #Data Minimum And Max Value


# In[112]:


#The correlation matrix is a square matrix that shows how strongly pairs of variables are related to each other.
df.corr()


# In[113]:


df.info()


# In[114]:


df.isnull().sum()  #Checking null value


# In[115]:


#Or other function to check null value
df.isna().sum()


# In[116]:


df.isnull().values.any() #IF there any null value reture True otherwise false


# In[117]:


#Delete null value from row 
#df = df.dropna()   then df.head()


# In[118]:


# Multiple columns remove= df.drop(columns = ["slope","ca"],axis = 1) 
# df.head()

#Single Column Delete
# df = df.drop(['thal'],axis = 1) 

#axis = 1 means delete column null value, axis = 0 means delete column from row  then df.head()


# ### Data Visualization

# In[119]:


#sn.countplot(df["target"])  
#for visualize


# In[120]:


#gender (1 = male; 0 = female) 
# 0 = no disease, 1 = disease

sn.countplot(x='gender' ,hue='target' ,data=df,palette='colorblind' ,edgecolor=sn.color_palette('dark',n_colors=1))


# In[121]:


# normal attribute and class attribute feature alada korar jonno
x = df.iloc[:,:-1]


# In[122]:


x


# In[123]:


y = df.iloc[:,13]


# In[124]:


y


# ### Splitting Data

# In[125]:


# One will use for train and one will use for testing

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state= 137)  #20% for testing


# In[126]:


len(x_train), len(x_test)


# # Random Forest Classifier 

# In[127]:


rf_clf = RandomForestClassifier(random_state=137) # For reproducibility
rf_clf.fit(x_train,y_train)


# In[128]:


rf_pred = rf_clf.predict(x_test) # Make predictions on the test data


# In[129]:


print("Random Forest MSE:",mean_squared_error(y_test,rf_pred))
print("Random Forest Accuracy:",accuracy_score(y_test,rf_pred))
print("Random Forest Precision:",precision_score(y_test,rf_pred))
print("Random Forest Recall:",recall_score(y_test,rf_pred))
print("Random Forest F1 Score:",f1_score(y_test,rf_pred))
print("Random Forest Confusion Matrix:",confusion_matrix(y_test,rf_pred))


# In[130]:


# Another Way to find Accuuracy
rf_clf.score(x_test, y_test)


# # Decision Tree Classifier

# In[131]:


dc_clf = DecisionTreeClassifier(random_state=137)
dc_clf.fit(x_train,y_train) 


# In[132]:


dc_pred = dc_clf.predict(x_test) # Make predictions on the test data


# In[133]:


print("Decision Tree MSE:",mean_squared_error(y_test,dc_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test,dc_pred))
print("Decision Tree Precision:",precision_score(y_test,dc_pred))
print("Decision Tree Recall:",recall_score(y_test,dc_pred))
print("Decision Tree F1 Score:",f1_score(y_test,dc_pred))
print("Decision Tree Confusion Matrix:", confusion_matrix(y_test,dc_pred))


# In[134]:


# Another Way to find Accuuracy
dc_clf.score(x_test, y_test)


# # Naive Bayes

# In[135]:


nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)


# In[136]:


nb_pred = nb_clf.predict(x_test)


# In[137]:


print("Naive Bayes MSE:",mean_squared_error(y_test,nb_pred))
print("Naive Bayes Accuracy:",accuracy_score(y_test,nb_pred))
print("Naive Bayes Precision:",precision_score(y_test,nb_pred))
print("Naive Bayes Recall:",recall_score(y_test,dc_pred))
print("Naive Bayes F1 Score:",f1_score(y_test,dc_pred))
print("Naive Bayes Confusion Matrix:",confusion_matrix(y_test,nb_pred))

