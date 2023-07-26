#!/usr/bin/env python
# coding: utf-8

# # Heart Disease

# ### Import Libraries

# In[975]:


import pandas as pd #Read file
import numpy as np #Matrix multiplication
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, confusion_matrix


# ### Load Dataset

# In[922]:


df = pd.read_csv(r"C:\Users\Khadiza\Downloads\Heart Attack Data Set.csv", na_values ='?')


# In[923]:


df.head(20)  #Top 5 row


# In[924]:


df.tail()


# ### Info of Dataset

# In[925]:


df.info()


# In[926]:


df.isnull().sum()  #Checking null value


# In[927]:


df.describe() #Data Minimum And Max Value


# In[928]:


#df.drop(columns = ["slope","ca"],axis = 1) axis = 1 means delete column null value, axis = 0 means delete column from row


# In[929]:


#Delete null value from row df = df.dropna()   then df.head()


# In[930]:


df["ca"].value_counts()  # 1 = male; 0 = female  //Numerical value , unique value


# ### Dummies using pandas to convert categorical value to one-hot encoding

# In[931]:


df = pd.get_dummies(df,columns = ["cp","restecg","slope","thal"])


# In[932]:


df.head()


# In[933]:


#How to see all columns df.columns


# In[934]:


# Rename columns name  df = df.rename(columns = {"target":"output"})


# In[935]:


numerical_cols = ["age","trestbps","chol","thalach","oldpeak"]
cat_cols = list(set(df.columns) - set(numerical_cols) -{"target"})


# In[936]:


cat_cols


# In[937]:


numerical_cols


# ### Splitting Data

# In[938]:


# One will use for train and one will use for testing


# In[939]:


df_train , df_test = train_test_split(df,test_size = 0.2 ,random_state= 42)  #20% for testing


# In[940]:


len(df_train) , len(df_test)


# In[941]:


# Create a StandardScaler object
scaler = StandardScaler()

def get_features_and_target_arrays(df,numerical_cols,cat_cols,scaler):
    x_numeric_scaled = scaler.fit_transform(df[numerical_cols]) # Numerical comlumns
    x_categorical = df[cat_cols].to_numpy()  # Categoricall comlumns
    x = np.hstack((x_categorical,x_numeric_scaled))
    y = df["target"]
    
    return x,y


# In[942]:


x_train , y_train = get_features_and_target_arrays(df_train,cat_cols,numerical_cols,scaler)


# In[943]:


### Create and train Logistic Regression Model

logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)


# In[944]:


x_test , y_test = get_features_and_target_arrays(df_test,cat_cols,numerical_cols,scaler)


# In[945]:


test_pred = logistic_model.predict(x_test) # Make predictions on the test data


# In[946]:


print("Logistic Regression MSE:", mean_squared_error(y_test,test_pred))


# In[947]:


print("Logistic Regression Accuracy:",accuracy_score(y_test,test_pred))


# In[948]:


print("Logistic Regression Precision:",precision_score(y_test,test_pred))


# In[949]:


print("Logistic Regression Confusion Matrix:", confusion_matrix(y_test,test_pred))


# ### Decision Tree

# In[950]:


# Desicion Tree Classifier
dc_clf = DecisionTreeClassifier()
dc_clf.fit(x_train,y_train) 


# In[968]:


dc_pred = dc_clf.predict(x_test)


# In[969]:


print("Decision Tree MSE:",mean_squared_error(y_test,dc_pred))


# In[974]:


print("Decision Tree Accuracy:", accuracy_score(y_test,dc_pred))


# In[971]:


print("Decision Tree Precision:",precision_score(y_test,dc_pred))


# In[955]:


print("Decision Tree Confusion Matrix:", confusion_matrix(y_test,dc_pred))


# ### Random Forest

# In[956]:


rf_clf = RandomForestClassifier()
rf_clf.fit(x_train,y_train)


# In[957]:


rf_pred = rf_clf.predict(x_test)


# In[958]:


print("Random Forest MSE:",mean_squared_error(y_test,rf_pred))


# In[959]:


print("Random Forest Accuracy:",accuracy_score(y_test,rf_pred))


# In[960]:


print("Random Forest Precision:",precision_score(y_test,rf_pred))


# In[961]:


print("Random Forest Confusion Matrix:", confusion_matrix(y_test,rf_pred))


# ### Naive Bayes

# In[962]:


nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)


# In[963]:


nb_pred = nb_clf.predict(x_test)


# In[964]:


print("Naive Bayes MSE:",mean_squared_error(y_test,nb_pred))


# In[965]:


print("Naive Bayes Accuracy:",accuracy_score(y_test,nb_pred))


# In[966]:


print("Naive Bayes Precision:",precision_score(y_test,nb_pred))


# In[967]:


print("Naive Bayes Confusion Matrix:", confusion_matrix(y_test,nb_pred))


# In[ ]:




