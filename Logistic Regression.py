#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 00:07:56 2018

@author: jayendra
"""

import numpy as np
import pandas as pd
import seaborn as sb 
%matplotlib inline
import matplotlib.pyplot as plt import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression from sklearn.cross_validation import train_test_split from sklearn import metrics
from sklearn.metrics import classification_report

link = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
data = pd.read_csv(link)
data.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
data.head()

type(data)

data.isnull().sum()

data = data.drop(['PassengerId','Name','Ticket','Cabin'], 1)
data.head()

sb.boxplot(x='Pclass', y='Age', data=data, palette='hls')

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
data['Age'] = data[['Age', 'Pclass']].apply(age_approx, axis=1)
data.isnull().sum()

data.dropna(inplace=True)
data.isnull().sum()

#Converting categorical variables to a dummy variables
gender = pd.get_dummies(data['Sex'],drop_first=True)
gender.head()
embark_location = pd.get_dummies(data['Embarked'],drop_first=True)
embark_location.head()
data.head()
data.drop(['Sex', 'Embarked'],axis=1,inplace=True)
data.head()
data_dmy = pd.concat([data,gender,embark_location],axis=1)
data_dmy.head()
data_dmy.corr()  
data_dmy.drop(['Fare', 'Pclass'],axis=1,inplace=True)
data_dmy.head()
X = data_dmy.ix[:,(1,2,3,4,5,6)].values
y = data_dmy.ix[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split,KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge

C_parameter =[0.0001,0.0002,0.0005,0.001,0.005,0.01,0.05, 0.1,0.5, 1, 10, 100,500,1000] 
auc_cv=[]
for a in C_parameter:
    logr=LogisticRegression(C=a,penalty="l1",class_weight="balanced",random_state=2)
    k = KFold(len(X_train), n_folds=10)
    score=0
    for train, test in k:
        logr.fit(X_train[train], y_train[train])
        score+=roc_auc_score(y_train,logr.predict(X_train))
    auc_cv.append(score/10)
    print('{:.3f}\t {:.5f}\t '.format(a,score/10))
C_parameter=np.array(C_parameter)
auc_cv=np.array(auc_cv)
c_best=C_parameter[auc_cv==max(auc_cv)][0]
print("The Value of C Best=",c_best)
y_pred= logr.predict(X_test)
from sklearn.metrics import confusion_matrix
cmp = confusion_matrix(y_test, y_pred)
print("Confusion Matrix \n",cmp)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % max(auc_cv))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()
print("AUC Ridge = ",max(auc_cv)*100)

C_parameter =[0.0001,0.0002,0.0005,0.001,0.005,0.01,0.05, 0.1,0.5, 1, 10, 100,500,1000] 
auc_cv_1=[]
for a in C_parameter:
    logreg=LogisticRegression(C=a,penalty="l2",class_weight="balanced",random_state=2)
    k = KFold(len(X_train), n_folds=10)
    score=0
    for train, test in k:
        logreg.fit(X_train[train], y_train[train])
        score+=roc_auc_score(y_train,logreg.predict(X_train))
    auc_cv_1.append(score/10)
    print('{:.3f}\t {:.5f}\t '.format(a,score/10))
C_parameter=np.array(C_parameter)
auc_cv_1=np.array(auc_cv_1)
c_best=C_parameter[auc_cv_1==max(auc_cv_1)][0]
print("The Value of C Best=",c_best)
y_pred= logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
cmp = confusion_matrix(y_test, y_pred)
print("Confusion Matrix \n",cmp)


from sklearn.metrics import roc_auc_score
from sklearn.metrics import  roc_curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % max(auc_cv_1))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()
print("AUC Lasso = ",max(auc_cv_1)*100)


print(classification_report(y_test, y_pred))

import pandas as pd
data = [['AUC Ridge ',max(auc_cv)*100],['AUC Lasso ',max(auc_cv_1)*100]]
df = pd.DataFrame(data,columns=['Loss Method','Accuracy'])
print(df)

