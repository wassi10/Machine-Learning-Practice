# -*- coding: utf-8 -*-
"""Heart Disease Prediction Using Machine Learning Algorithms.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yzDd9T58pW1KGy1rzbMquF6Q0fN8XT-_

# **Heart Disease Prediction**

### **ML Algorithms: LR, RF, DT, NB, SVM, GB, MLP, KNN, XGBoost**

### **Import Libraries**
"""

import pandas as pd #Read file
import numpy as np #Matrix multiplication
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay

df = pd.read_csv('/content/heart.csv')

df.shape

df.head(10)

"""**Info Of Dataset**"""

df.isnull().values.any() #IF there any null value reture True otherwise false

# normal attribute and class attribute feature alada korar jonno
x = df.iloc[:,:-1]

x

correlation=df.corr()
correlation

#heatmap
plt.figure(figsize=(14,5))
sns.heatmap(correlation, cmap="Greens", annot=True)

y = df.iloc[:,13]

y

# One will use for train and one will use for testing

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1 ,random_state= 1)  #25% for testing

sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

"""**RF**"""

param_grid_rf = {
'n_estimators': [50, 100],  # Reduced number of estimators
    'max_depth': [5],  # Increased max_depth or set it to a specific value
    'min_samples_split': [2, 5, 10],  # Increased min_samples_split
    'min_samples_leaf': [1],  # Increased min_samples_leaf
    'max_features': ['auto', 'sqrt', 0.5],
    'random_state': [0]
}
rf = RandomForestClassifier()
rf_clf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='accuracy')
rf_clf.fit(x_train,y_train)

rf_pred = rf_clf.predict(x_test) # Make predictions on the test data

print("Random Forest Accuracy:",accuracy_score(y_test,rf_pred))
print("Random Forest Precision:",precision_score(y_test,rf_pred))
print("Random Forest Recall:",recall_score(y_test,rf_pred))
print("Random Forest F1 Score:",f1_score(y_test,rf_pred))
print("Random Forest Confusion Matrix:",confusion_matrix(y_test,rf_pred))

"""**DT**"""

dc_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
dc_clf.fit(x_train,y_train)

dc_pred = dc_clf.predict(x_test) # Make predictions on the test data

print("Decision Tree Accuracy:", accuracy_score(y_test,dc_pred))
print("Decision Tree Precision:",precision_score(y_test,dc_pred))
print("Decision Tree Recall:",recall_score(y_test,dc_pred))
print("Decision Tree F1 Score:",f1_score(y_test,dc_pred))
print("Decision Tree Confusion Matrix:", confusion_matrix(y_test,dc_pred))

"""**NB**"""

nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)

nb_pred = nb_clf.predict(x_test)

print("Naive Bayes Accuracy:",accuracy_score(y_test,nb_pred))
print("Naive Bayes Precision:",precision_score(y_test,nb_pred))
print("Naive Bayes Recall:",recall_score(y_test,dc_pred))
print("Naive Bayes F1 Score:",f1_score(y_test,dc_pred))
print("Naive Bayes Confusion Matrix:",confusion_matrix(y_test,nb_pred))

"""**Logistic Regression**"""

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter':[100,1000,10000]
}
logreg = LogisticRegression()
model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, n_jobs=-1)

model.fit(x_train,y_train)

ans=model.predict(x_train)

accuracy=accuracy_score(ans,y_train)
precision=precision_score(ans,y_train)
recall=recall_score(ans,y_train)
f1=f1_score(ans,y_train)

print("Training Accuracy is ",accuracy)
print("Training Precisiom is ",precision)
print("Training recall is ",recall)
print("Training F1 is ",f1)

anss=model.predict(x_test)
accuracy_test_lr=accuracy_score(anss,y_test)
precision_test_lr=precision_score(anss,y_test)
recall_test_lr=recall_score(anss,y_test)
f1_test_lr=f1_score(anss,y_test)
print("Testing Accuracy is ",accuracy_test_lr)
print("Testing Precisiom is ",precision_test_lr)
print("Testing recall is ",recall_test_lr)
print("Testing F1 is ",f1)

"""**XGBoost**"""

# Create and train the XGBoost model with normalized data
model2 = xgb.XGBClassifier(reg_alpha=2.0, max_depth=2, min_child_weight=5.0,min_samples_leaf= 2,random_state=0)
model2.fit(x_train, y_train)

acc = model2.predict(x_train)
accuracy_xgb=accuracy_score(acc,y_train)
precision_xgb=precision_score(acc,y_train)
recall_xgb=recall_score(acc,y_train)
f1_xgb=f1_score(acc,y_train)
print("Training Accuracy is ",accuracy_xgb)
print("Training Precisiom is ",precision_xgb)
print("Training recall is ",recall_xgb)
print("Training F1 is ",f1_xgb)

acc2 = model2.predict(x_test)
accuracy_xgb_test=accuracy_score(acc2,y_test)
precision_xgb_test=precision_score(acc2,y_test)
recall_xgb_test=recall_score(acc2,y_test)
f1_xgb_test=f1_score(acc2,y_test)
print("Testing Accuracy is ",accuracy_xgb_test)
print("Testing Precisiom is ",precision_xgb_test)
print("Testing recall is ",recall_xgb_test)
print("Testing F1 is ",f1_xgb_test)

"""# MLPClassifier"""

model3 = MLPClassifier(hidden_layer_sizes=
                       (50,),random_state=1)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_train)
acc_mlp  = accuracy_score(y_pred,y_train)
precision_mlp=precision_score(y_pred,y_train)
recall_mlp=recall_score(y_pred,y_train)
f1_mlp=f1_score(y_pred,y_train)
print("Traing Accuracy is ",acc_mlp)
print("Traing Precisiom is ",precision_mlp)
print("Traing recall is ",recall_mlp)
print("Traing F1 is ",f1_mlp)

y_pred_t = model3.predict(x_test)
acc_mlp_t  = accuracy_score(y_pred_t,y_test)
precision_mlp_t=precision_score(y_pred_t,y_test)
recall_mlp_t=recall_score(y_pred_t,y_test)
f1_mlp_t=f1_score(y_pred_t,y_test)
print("Testing Accuracy is ",acc_mlp_t)
print("Testing Precisiom is ",precision_mlp_t)
print("Testing recall is ",recall_mlp_t)
print("Testing F1 is ",f1_mlp_t)

"""**SVM**"""

classifier = svm.SVC(kernel='linear', gamma='auto',C=2)
classifier.fit(x_train,y_train)

Y_predict = classifier.predict(x_test)

Y_predict = classifier.predict(x_test)

print("SVM Accuracy:",accuracy_score(y_test,Y_predict))
print("SVM Precision:",precision_score(y_test,Y_predict))
print("SVM Forest Recall:",recall_score(y_test,Y_predict))
print("SVM Forest F1 Score:",f1_score(y_test,Y_predict))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Y_predict)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

"""# KNN

"""

#When K = 5

#KNN MODEL building


knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier = knn_classifier.fit(x_train,y_train)


#prediction

Y_pred = knn_classifier.predict(x_test)

print("KNN Accuracy:",accuracy_score(y_test,Y_pred))
print("KNN Precision:",precision_score(y_test,Y_pred))
print("KNN Forest Recall:",recall_score(y_test,Y_pred))
print("KNN Forest F1 Score:",f1_score(y_test,Y_pred))

#when k = 6

knn_classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train,y_train)

#prediction
Y_pred = knn_classifier.predict(x_test)

print("KNN Accuracy:",accuracy_score(y_test,Y_pred))
print("KNN Precision:",precision_score(y_test,Y_pred))
print("KNN Forest Recall:",recall_score(y_test,Y_pred))
print("KNN Forest F1 Score:",f1_score(y_test,Y_pred))

# When k= 7

knn_classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train,y_train)
#prediction
Y_pred = knn_classifier.predict(x_test)
print("KNN Accuracy:",accuracy_score(y_test,Y_pred))
print("KNN Precision:",precision_score(y_test,Y_pred))
print("KNN Forest Recall:",recall_score(y_test,Y_pred))
print("KNN Forest F1 Score:",f1_score(y_test,Y_pred))

#When k= 8


knn_classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train,y_train)

#prediction
Y_pred = knn_classifier.predict(x_test)

#check accuracy

print("KNN Accuracy:",accuracy_score(y_test,Y_pred))
print("KNN Precision:",precision_score(y_test,Y_pred))
print("KNN Forest Recall:",recall_score(y_test,Y_pred))
print("KNN Forest F1 Score:",f1_score(y_test,Y_pred))

#When k = 9


knn_classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train,y_train)

#prediction
Y_pred = knn_classifier.predict(x_test)

print("KNN Accuracy:",accuracy_score(y_test,Y_pred))
print("KNN Precision:",precision_score(y_test,Y_pred))
print("KNN Forest Recall:",recall_score(y_test,Y_pred))
print("KNN Forest F1 Score:",f1_score(y_test,Y_pred))

cm = confusion_matrix(y_test,Y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_classifier.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()



"""**GB Classifier implementation**"""

# Create the Gradient Boosting classifier
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.001],
    'max_depth': [5],
    'min_samples_split': [30],
    'min_samples_leaf': [1],
    'random_state': [0]
}

# Instantiate the GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb_classifier = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)

# Train the classifier on the training data
gb_classifier.fit(x_train, y_train)

# Make predictions on the test data
Y_pred2 = gb_classifier.predict(x_test)

print("GB Accuracy:",accuracy_score(y_test,Y_pred2))
print("GB Precision:",precision_score(y_test,Y_pred2))
print("GB Forest Recall:",recall_score(y_test,Y_pred2))
print("GB Forest F1 Score:",f1_score(y_test,Y_pred2))

#confusion matrix

cm = confusion_matrix(y_test,Y_pred2)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gb_classifier.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

final_data = pd.DataFrame({'Models':['LR','RF','NB','DT','SVM','KNN','GB','MLP','XGB'],'ACC':[
    accuracy_score(anss,y_test),
    accuracy_score(y_test,rf_pred),
    accuracy_score(y_test,nb_pred),
    accuracy_score(y_test,dc_pred),
   accuracy_score(y_test, Y_predict),
   accuracy_score(y_test, Y_pred),
   accuracy_score(y_test, Y_pred2),
  accuracy_score(y_pred_t,y_test),
  accuracy_score(acc2,y_test)

                                                              ]})

final_data

# Plotting the bar chart using Seaborn
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.barplot(x='Models', y='ACC', data=final_data)
plt.title('Model Accuracy Comparison')
plt.xlabel('9 Models')
plt.ylabel('Accuracy')
for index, row in final_data.iterrows():
    plt.text(index, row['ACC'] + 0.01, f'{row["ACC"]*100:.2f}', ha='center', fontsize=12)
plt.show()

