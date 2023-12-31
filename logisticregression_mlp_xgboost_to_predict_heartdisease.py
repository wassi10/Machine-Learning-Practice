# -*- coding: utf-8 -*-
"""LogisticRegression_MLP_XGBoost to predict HeartDisease.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jWBz7-g3ysvYP_1iTIzLUtsZMN77UYHt

importing libreries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler





from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""Addding data into dataframe"""

data = pd.read_csv('/content/Heart_Attack_Data_Set.csv')
data.head()

correlation = data.corr()
correlation
#heatmap
plt.figure(figsize=(18,8))
sns.heatmap(correlation, cmap="Blues", annot=True)

"""Checking rows and columns"""

data.shape

"""counting the count of disease and no disease"""

data['target'].value_counts()

"""Dividing the outcome and input"""

X=data.drop(columns='target',axis=1)
Y=data['target']

X

Y

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_train

print(X.shape,X_train.shape,X_test.shape)

"""#LogisticRegression"""

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter':[100,1000,10000]
}
logreg = LogisticRegression(max_iter=1000)
model= GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, n_jobs=-1)
model.fit(X_train,Y_train)
ans=model.predict(X_train)
anss=model.predict(X_test)
accuracy_test_lr=accuracy_score(anss,Y_test)
precision_test_lr=precision_score(anss,Y_test)
recall_test_lr=recall_score(anss,Y_test)
f1_test_lr=f1_score(anss,Y_test)
print("Testing Accuracy is ",accuracy_test_lr)
print("Testing Precisiom is ",precision_test_lr)
print("Testing recall is ",recall_test_lr)
print("Testing F1 is ",f1_test_lr)

"""# LogisticRegression
#Result:
1.   Accuracy 90.32%
2.   Precision 88.23
3.Recall 93.75
4.F1 score 86.26

#XGBoost
"""

model2=model = xgb.XGBClassifier(reg_alpha=2.0, max_depth=2, min_child_weight=5.0)
model2.fit(X_train,Y_train)
acc22 = model2.predict(X_test)
accuracy_xgb_test=accuracy_score(acc22,Y_test)
precision_xgb_test=precision_score(acc22,Y_test)
recall_xgb_test=recall_score(acc22,Y_test)
f1_xgb_test=f1_score(acc22,Y_test)
print("Testing Accuracy is ",accuracy_xgb_test)
print("Testing Precisiom is ",precision_xgb_test)
print("Testing recall is ",recall_xgb_test)
print("Testing F1 is ",f1_xgb_test)

"""#XGBoost
#Result :
1.   Accuracy is 74.19%
2.  Precision 64.70
3. Recall :84.61
4.F1:73.33

data.isnull()
"""

data.isnull().sum()

"""#MLPClassifier"""

model3 = MLPClassifier(hidden_layer_sizes=(50,),random_state=1)
model3.fit(X_train, Y_train)
y_pred_t = model3.predict(X_test)
acc_mlp_t  = accuracy_score(y_pred_t,Y_test)
precision_mlp_t=precision_score(y_pred_t,Y_test)
recall_mlp_t=recall_score(y_pred_t,Y_test)
f1_mlp_t=f1_score(y_pred_t,Y_test)
print("Testing Accuracy is ",acc_mlp_t)
print("Testing Precisiom is ",precision_mlp_t)
print("Testing recall is ",recall_mlp_t)
print("Testing F1 is ",f1_mlp_t)

"""#MLPClassifier
#Result:

1.   Acuuracy:90.32
2.   Precision:94.11
3.Recall:88.88
4.F1 :91.42

#Priya's Algorithm Starts from here
"""



"""# **Random Forest, Decision Tree and Naive Bayes**

**Random Forest**
"""

rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train,Y_train)
rf_pred = rf_clf.predict(X_test)
print("Random Forest Accuracy:",accuracy_score(Y_test,rf_pred))
print("Random Forest Precision:",precision_score(Y_test,rf_pred))
print("Random Forest Recall:",recall_score(Y_test,rf_pred))
print("Random Forest F1 Score:",f1_score(Y_test,rf_pred))
print("Random Forest Confusion Matrix:",confusion_matrix(Y_test,rf_pred))

"""**Decison Tree**"""

dc_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
dc_clf.fit(X_train,Y_train)
dc_pred = dc_clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(Y_test,dc_pred))
print("Decision Tree Precision:",precision_score(Y_test,dc_pred))
print("Decision Tree Recall:",recall_score(Y_test,dc_pred))
print("Decision Tree F1 Score:",f1_score(Y_test,dc_pred))
print("Decision Tree Confusion Matrix:", confusion_matrix(Y_test,dc_pred))

"""**Naive Bayes**"""

nb_clf = GaussianNB()
nb_clf.fit(X_train,Y_train)
nb_pred_22= nb_clf.predict(X_test)
print("Naive Bayes Accuracy:",accuracy_score(Y_test,nb_pred_22))
print("Naive Bayes Precision:",precision_score(Y_test,nb_pred_22))
print("Naive Bayes Recall:",recall_score(Y_test,nb_pred_22))
print("Naive Bayes F1 Score:",f1_score(Y_test,nb_pred_22))
print("Naive Bayes Confusion Matrix:",confusion_matrix(Y_test,nb_pred_22))

"""**Support Vector Machine**"""

classifier = svm.SVC(kernel='linear', gamma='auto',C=2)
classifier.fit(X_train,Y_train)
Y_predict = classifier.predict(X_test)
print("svm Accuracy:", accuracy_score(Y_test,Y_predict))
print("svm Precision:",precision_score(Y_test,Y_predict))
print("svm Recall:",recall_score(Y_test,Y_predict))
print("svm F1 Score:",f1_score(Y_test,Y_predict))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_predict)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

"""**KNN Algorithm implementation**"""

#When K = 5

#KNN MODEL building


knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier = knn_classifier.fit(X_train,Y_train)


#prediction

Y_pred = knn_classifier.predict(X_test)

print("KNN Accuracy:", accuracy_score(Y_test,Y_pred))
print("KNN Precision:",precision_score(Y_test,Y_pred))
print("KNN Recall:",recall_score(Y_test,Y_pred))
print("KNN F1 Score:",f1_score(Y_test,Y_pred))

#when k = 6

knn_classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train,Y_train)

#prediction
Y_pred = knn_classifier.predict(X_test)

print("KNN Accuracy:", accuracy_score(Y_test,Y_pred))
print("KNN Precision:",precision_score(Y_test,Y_pred))
print("KNN Recall:",recall_score(Y_test,Y_pred))
print("KNN F1 Score:",f1_score(Y_test,Y_pred))

# When k= 7

knn_classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train,Y_train)
#prediction
Y_pred = knn_classifier.predict(X_test)

print("KNN Accuracy:", accuracy_score(Y_test,Y_pred))
print("KNN Precision:",precision_score(Y_test,Y_pred))
print("KNN Recall:",recall_score(Y_test,Y_pred))
print("KNN F1 Score:",f1_score(Y_test,Y_pred))

#When k= 8


knn_classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train,Y_train)

#prediction
Y_pred = knn_classifier.predict(X_test)

#check accuracy

print("KNN Accuracy:", accuracy_score(Y_test,Y_pred))
print("KNN Precision:",precision_score(Y_test,Y_pred))
print("KNN Recall:",recall_score(Y_test,Y_pred))
print("KNN F1 Score:",f1_score(Y_test,Y_pred))

#When k = 9


knn_classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train,Y_train)

#prediction
Y_pred = knn_classifier.predict(X_test)

print("KNN Accuracy:", accuracy_score(Y_test,Y_pred))
print("KNN Precision:",precision_score(Y_test,Y_pred))
print("KNN Recall:",recall_score(Y_test,Y_pred))
print("KNN F1 Score:",f1_score(Y_test,Y_pred))

cm = confusion_matrix(Y_test,Y_pred)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_classifier.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

"""# GB Classifier implementation"""

# Create the Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=2)

# Train the classifier on the training data
gb_classifier.fit(X_train, Y_train)
Y_pred2 = gb_classifier.predict(X_test)
print("GB Accuracy:", accuracy_score(Y_test,Y_pred2))
print("GB Precision:",precision_score(Y_test,Y_pred2))
print("GB Recall:",recall_score(Y_test,Y_pred2))
print("GB F1 Score:",f1_score(Y_test,Y_pred2))
print("GB Confusion Matrix:", confusion_matrix(Y_test,Y_pred2))

#confusion matrix

cm = confusion_matrix(Y_test,Y_pred2)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gb_classifier.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

final_data = pd.DataFrame({'Models':['LR','RF','NB','DT','SVM','KNN','GB','MLP','XGB'],'ACC':[
     accuracy_score(anss,Y_test),
      accuracy_score(Y_test,rf_pred),
       accuracy_score(Y_test,nb_pred_22),
        accuracy_score(Y_test,dc_pred),
     accuracy_score(Y_test,Y_predict),
     accuracy_score(Y_test,Y_pred),
     accuracy_score(Y_test,Y_pred2),
     accuracy_score(y_pred_t,Y_test),
     accuracy_score(acc22,Y_test)

                                                                                    ]})

final_data

# Plotting the bar chart using Seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.barplot(x='Models', y='ACC', data=final_data)
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('9 Models', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
for index, row in final_data.iterrows():
    plt.text(index, row['ACC'] + 0.01, f'{row["ACC"]*100:.2f}', ha='center', fontsize=12)
plt.show()

"""# *DATASET-02*"""

df = pd.read_csv('/content/heart.csv')
x = df.iloc[:,:-1]
correlation=df.corr()
plt.figure(figsize=(18,8))
sns.heatmap(correlation, cmap="Purples", annot=True)

y = df.iloc[:,13]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1 ,random_state= 1)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

"""#RF"""

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
rf_pred_2 = rf_clf.predict(x_test)

print("Random Forest Accuracy:",accuracy_score(y_test,rf_pred_2))
print("Random Forest Precision:",precision_score(y_test,rf_pred_2))
print("Random Forest Recall:",recall_score(y_test,rf_pred_2))
print("Random Forest F1 Score:",f1_score(y_test,rf_pred_2))
print("Random Forest Confusion Matrix:",confusion_matrix(y_test,rf_pred_2))

"""#DT"""

dc_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
dc_clf.fit(x_train,y_train)
dc_pred_2 = dc_clf.predict(x_test)
print("Decision Tree Accuracy:", accuracy_score(y_test,dc_pred_2))
print("Decision Tree Precision:",precision_score(y_test,dc_pred_2))
print("Decision Tree Recall:",recall_score(y_test,dc_pred_2))
print("Decision Tree F1 Score:",f1_score(y_test,dc_pred_2))
print("Decision Tree Confusion Matrix:", confusion_matrix(y_test,dc_pred_2))

"""#NB"""

nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)
nb_pred = nb_clf.predict(x_test)
print("Naive Bayes Accuracy:",accuracy_score(y_test,nb_pred))
print("Naive Bayes Precision:",precision_score(y_test,nb_pred))
print("Naive Bayes Recall:",recall_score(y_test,nb_pred))
print("Naive Bayes F1 Score:",f1_score(y_test,nb_pred))
print("Naive Bayes Confusion Matrix:",confusion_matrix(y_test,nb_pred))

"""#LR"""

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter':[100,1000,10000]
}
logreg = LogisticRegression()
model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, n_jobs=-1)
model.fit(x_train,y_train)
anss_22=model.predict(x_test)
accuracy_test_lr=accuracy_score(anss_22,y_test)
precision_test_lr=precision_score(anss_22,y_test)
recall_test_lr=recall_score(anss_22,y_test)
f1_test_lr=f1_score(anss_22,y_test)
print("Testing Accuracy is ",accuracy_test_lr)
print("Testing Precisiom is ",precision_test_lr)
print("Testing recall is ",recall_test_lr)
print("Testing F1 is ",f1_test_lr)

"""#XGB"""

model2 = xgb.XGBClassifier(reg_alpha=2.0, max_depth=2, min_child_weight=5.0,min_samples_leaf= 2,random_state=0)
model2.fit(x_train, y_train)
acc2_2 = model2.predict(x_test)
print("Testing Accuracy is ",accuracy_score(acc2_2,y_test))
print("Testing Precisiom is ",precision_score(acc2_2,y_test))
print("Testing recall is ",recall_score(acc2_2,y_test))
print("Testing F1 is ",f1_score(acc2_2,y_test))

"""#MLP"""

model3 = MLPClassifier(hidden_layer_sizes=
                       (50,),random_state=1)
model3.fit(x_train, y_train)
y_pred_t_2 = model3.predict(x_test)
acc_mlp_t  = accuracy_score(y_pred_t_2,y_test)
precision_mlp_t=precision_score(y_pred_t_2,y_test)
recall_mlp_t=recall_score(y_pred_t_2,y_test)
f1_mlp_t=f1_score(y_pred_t_2,y_test)
print("Testing Accuracy is ",acc_mlp_t)
print("Testing Precisiom is ",precision_mlp_t)
print("Testing recall is ",recall_mlp_t)
print("Testing F1 is ",f1_mlp_t)

"""#SVM"""

classifier = svm.SVC(kernel='linear', gamma='auto',C=2)
classifier.fit(x_train,y_train)

Y_predict_2 = classifier.predict(x_test)
print("SVM Accuracy:",accuracy_score(y_test,Y_predict_2))
print("SVM Precision:",precision_score(y_test,Y_predict_2))
print("SVM Forest Recall:",recall_score(y_test,Y_predict_2))
print("SVM Forest F1 Score:",f1_score(y_test,Y_predict_2))

"""#KNN"""

knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier = knn_classifier.fit(x_train,y_train)


#prediction

Y_pred_2 = knn_classifier.predict(x_test)

print("KNN Accuracy:",accuracy_score(y_test,Y_pred_2))
print("KNN Precision:",precision_score(y_test,Y_pred_2))
print("KNN Forest Recall:",recall_score(y_test,Y_pred_2))
print("KNN Forest F1 Score:",f1_score(y_test,Y_pred_2))

"""#GB"""

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
Y_pred2_2 = gb_classifier.predict(x_test)
print("GB Accuracy:",accuracy_score(y_test,Y_pred2_2))
print("GB Precision:",precision_score(y_test,Y_pred2_2))
print("GB Forest Recall:",recall_score(y_test,Y_pred2_2))
print("GB Forest F1 Score:",f1_score(y_test,Y_pred2_2))

"""Accuracy,Precision,Recall,F1"""

pip install seaborn

final_data = pd.DataFrame({'Models':['LR', 'MLP', 'XGB', 'RF', 'DT', 'SVM', 'KNN', 'GB','NB'],'ACC':[
     accuracy_score(anss_22,y_test),
                     accuracy_score(y_pred_t_2,y_test),
                     accuracy_score(acc2_2,y_test),
                     accuracy_score(y_test,rf_pred_2),
                     accuracy_score(y_test,dc_pred_2),
                     accuracy_score(y_test,Y_predict_2),
                     accuracy_score(y_test,Y_pred_2),
                     accuracy_score(y_test,Y_pred2_2),
                     accuracy_score(y_test,nb_pred),

                                                                                    ]})
final_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
# Add grid lines
# Define your model names and accuracy values for dataset 1 and dataset 2
model_names = ['LR', 'MLP', 'XGB', 'RF', 'DT', 'SVM', 'KNN', 'GB','NB']
accuracy_dataset1 = [accuracy_score(anss,Y_test),
                     accuracy_score(y_pred_t,Y_test),
                     accuracy_score(acc22,Y_test),
                     accuracy_score(Y_test,rf_pred),
                     accuracy_score(Y_test,dc_pred),
                     accuracy_score(Y_test,Y_predict),
                     accuracy_score(Y_test,Y_pred),accuracy_score(Y_test,Y_pred2),
                      accuracy_score(Y_test,nb_pred_22),
                     ]
accuracy_dataset2 = [accuracy_score(anss_22,y_test),
                     accuracy_score(y_pred_t_2,y_test),
                     accuracy_score(acc2_2,y_test),
                     accuracy_score(y_test,rf_pred_2),
                     accuracy_score(y_test,dc_pred_2),
                     accuracy_score(y_test,Y_predict_2),
                     accuracy_score(y_test,Y_pred_2),
                     accuracy_score(y_test,Y_pred2_2),
                     accuracy_score(y_test,nb_pred),
                     ]

# Set the positions for the groups
x = range(len(model_names))

# Create a larger figure
plt.figure(figsize=(10, 6),facecolor='white')
palette = sns.color_palette("deep")
# Create a bar plot

plt.bar([i - 0.2 for i in x], accuracy_dataset1, width=0.4, align='center', label='Dataset 1', color=palette[0], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)
plt.bar([i + 0.2 for i in x], accuracy_dataset2, width=0.4, align='center', label='Dataset 2', color=palette[1], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)


# Set the x-axis ticks and labels
plt.xticks(x, model_names, rotation=45, ha='right')
# Set the y-axis label
custom_y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1]
y_labels = [f'{int(val*100):d}' for val in custom_y_ticks]
plt.yticks(custom_y_ticks, y_labels)
plt.ylabel('Accuracy')

# Set the plot title
plt.title('Model Accuracy Comparison for Dataset 1 and Dataset 2')


plt.legend()

# Show the plot
plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
# Add grid lines
# Define your model names and accuracy values for dataset 1 and dataset 2
model_names = ['LR', 'MLP', 'XGB', 'RF', 'DT',  'SVM', 'KNN', 'GB','NB']

precision_dataset1 = [precision_score(anss,Y_test),
                      precision_score(y_pred_t,Y_test),
                      precision_score(acc22,Y_test),
                      precision_score(Y_test, rf_pred),
                      precision_score(Y_test, dc_pred),
                      precision_score(Y_test, Y_predict),
                      precision_score(Y_test, Y_pred),
                      precision_score(Y_test, Y_pred2),
                                            precision_score(Y_test, nb_pred_22),
                      ]

precision_dataset2 = [precision_score(anss_22, y_test),
                      precision_score(y_pred_t_2, y_test),
                      precision_score(acc2_2, y_test),
                      precision_score(y_test, rf_pred_2),
                      precision_score(y_test, dc_pred_2),
                      precision_score(y_test, Y_predict_2),
                      precision_score(y_test, Y_pred_2),
                     precision_score(y_test, Y_pred2_2),
                      precision_score(y_test, nb_pred),
                      ]


# Set the positions for the groups
x = range(len(model_names))

# Create a larger figure
plt.figure(figsize=(10, 6),facecolor='white')
palette = sns.color_palette("deep")
# Create a bar plot

plt.bar([i - 0.2 for i in x],precision_dataset1, width=0.4, align='center', label='Dataset 1', color=palette[5], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)
plt.bar([i + 0.2 for i in x], precision_dataset2, width=0.4, align='center', label='Dataset 2', color=palette[6], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)


# Set the x-axis ticks and labels
plt.xticks(x, model_names, rotation=45, ha='right')
# Set the y-axis label
custom_y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1]
y_labels = [f'{int(val*100):d}' for val in custom_y_ticks]
plt.yticks(custom_y_ticks, y_labels)
plt.ylabel('Precision')

# Set the plot title
plt.title('Models Precision Comparison for Dataset 1 and Dataset 2')



# Add a legend
plt.legend()
# Show the plot
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score

sns.set(style="whitegrid")
# Add grid lines
# Define your model names and recall values for dataset 1 and dataset 2
model_names = ['LR', 'MLP', 'XGB', 'RF', 'DT', 'NB', 'SVM', 'KNN', 'GB']

recall_dataset1 = [recall_score(anss, Y_test),
                   recall_score(y_pred_t, Y_test),
                   recall_score(acc22, Y_test),
                   recall_score(Y_test, rf_pred),
                   recall_score(Y_test, dc_pred),
                   recall_score(Y_test, nb_pred_22),
                   recall_score(Y_test, Y_predict),
                   recall_score(Y_test, Y_pred),
                   recall_score(Y_test, Y_pred2)]

recall_dataset2 = [recall_score(anss_22, y_test),
                   recall_score(y_pred_t_2, y_test),
                   recall_score(acc2_2, y_test),
                   recall_score(y_test, rf_pred_2),
                   recall_score(y_test, dc_pred_2),
                   recall_score(y_test, nb_pred),
                   recall_score(y_test, Y_predict_2),
                   recall_score(y_test, Y_pred_2),
                   recall_score(y_test, Y_pred2_2)]

# Set the positions for the groups
x = range(len(model_names))

# Create a larger figure
plt.figure(figsize=(10, 6), facecolor='white')
palette = sns.color_palette("deep")

# Create a bar plot
plt.bar([i - 0.2 for i in x], recall_dataset1, width=0.4, align='center', label='Dataset 1', color=palette[9], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)
plt.bar([i + 0.2 for i in x], recall_dataset2, width=0.4, align='center', label='Dataset 2', color=palette[3], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)

# Set the x-axis ticks and labels
plt.xticks(x, model_names, rotation=45, ha='right')

# Set the y-axis label
custom_y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y_labels = [f'{int(val*100):d}' for val in custom_y_ticks]
plt.yticks(custom_y_ticks, y_labels)
plt.ylabel('Recall')

# Set the plot title
plt.title('Model Recall Comparison for Dataset 1 and Dataset 2')



# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

sns.set(style="whitegrid")
# Add grid lines
# Define your model names and F1-score values for dataset 1 and dataset 2
model_names = ['LR', 'MLP', 'XGB', 'RF', 'DT', 'NB', 'SVM', 'KNN', 'GB']

f1_dataset1 = [f1_score(anss, Y_test),
               f1_score(y_pred_t, Y_test),
               f1_score(acc22, Y_test),
               f1_score(Y_test, rf_pred),
               f1_score(Y_test, dc_pred),
               f1_score(Y_test, nb_pred_22),
               f1_score(Y_test, Y_predict),
               f1_score(Y_test, Y_pred),
               f1_score(Y_test, Y_pred2)]

f1_dataset2 = [f1_score(anss_22, y_test),
               f1_score(y_pred_t_2, y_test),
               f1_score(acc2_2, y_test),
               f1_score(y_test, rf_pred_2),
               f1_score(y_test, dc_pred_2),
               f1_score(y_test, nb_pred),
               f1_score(y_test, Y_predict_2),
               f1_score(y_test, Y_pred_2),
               f1_score(y_test, Y_pred2_2)]

# Set the positions for the groups
x = range(len(model_names))

# Create a larger figure
plt.figure(figsize=(10, 6), facecolor='white')
palette = sns.color_palette("deep")

# Create a bar plot
plt.bar([i - 0.2 for i in x], f1_dataset1, width=0.4, align='center', label='Dataset 1', color=palette[4], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)
plt.bar([i + 0.2 for i in x], f1_dataset2, width=0.4, align='center', label='Dataset 2', color=palette[7], edgecolor='black', linewidth=1, alpha=0.8, zorder=2)

# Set the x-axis ticks and labels
plt.xticks(x, model_names, rotation=45, ha='right')

# Set the y-axis label
custom_y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y_labels = [f'{int(val*100):d}' for val in custom_y_ticks]
plt.yticks(custom_y_ticks, y_labels)
plt.ylabel('F1-Score')

# Set the plot title
plt.title('Model F1-Score Comparison for Dataset 1 and Dataset 2')


# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()