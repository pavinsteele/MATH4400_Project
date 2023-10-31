#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:29:56 2023

@author: charlessimon
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

import numpy as np
import seaborn as sns
import math
from sklearn import preprocessing

# metrics and cross val
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_score

#Smote
from imblearn.over_sampling import SMOTE

#operating system tools
import os
#for grid search and cv
from sklearn.pipeline import Pipeline
import sklearn.model_selection as skm
#scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#lda qda naive Bayes
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
#for log reg
from sklearn.linear_model import LogisticRegression
os.chdir('/Users/charlessimon/Documents/Comp data science bach/Math 4500/Group Project')
os.getcwd()

#label encoder object
label_encoder = preprocessing.LabelEncoder()

#get dataframe from separate csv's
df1 = pd.read_csv('application_record.csv', header = 0)
df2 = pd.read_csv('credit_record.csv', header = 0)


#creating database merged
dfPre = pd.get_dummies(df2['STATUS'], drop_first = False) 
dfPre = pd.concat([df2, dfPre], axis=1)
dfPre = dfPre.groupby('ID').agg({'MONTHS_BALANCE': 'min', '0': 'sum', '1': 'sum', '2': 'sum', '3': 'sum', '4': 'sum', '5': 'sum', 'C': 'sum', 'X' : 'sum'})
#merge

dfPre['MONTHS'] = (dfPre['MONTHS_BALANCE'] - 1) * -1

dfPre = pd.merge(df1, dfPre, on = 'ID')
dfPre.shape
#index: ID
#bool (Y/N or 1/0) variables: FLAG_OWN_CAR (Y/N), FLAG_OWN_REALTY (Y/N), FLAG_MOBIL (1/0), FLAG_WORK_PHONE (1/0), FLAG_PHONE (1/0), FLAG_EMAIL (1/0)
#numerical variables: AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED, XXYEARS_EMPLOYEDXX, XXCNT_CHILDRENXX, CNT_FAM_MEMBERS
#categorical variables: CODE_GENDER, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE


#finding our Acceptance Column
dfPre['PaidorNoLoan'] = (dfPre['C'] + dfPre['X'])/dfPre['MONTHS']
dfPre['OverTwoMonths'] = dfPre['2'] + dfPre['3'] + dfPre['4'] + dfPre['5']
#percentages
dfPre['PercentNoLoan'] = dfPre['X']/dfPre['MONTHS']
dfPre['PercentOnTime'] = dfPre['C']/dfPre['MONTHS']
dfPre['Percent0to1'] = dfPre['0']/dfPre['MONTHS']
dfPre['Percent1to2'] = dfPre['1']/dfPre['MONTHS']
dfPre['PercentOver2'] = dfPre['OverTwoMonths']/dfPre['MONTHS']

#All People
length1 = (len(dfPre))
#People we want:
#People who have less than eighty five percent overdue by one month (column 0)
filter1 = (dfPre['Percent0to1']) < .85
#People who have no more than five percent due by over 2 months (column 1)
filter2 = (dfPre['Percent1to2']) < .10
#People who have no overdue by more than two months (columns 2-5)
filter3 = (dfPre['PercentOver2']) == 0
#People who have no overdue by more than two months (columns 2-5)
filter4 = dfPre['PercentNoLoan'] != dfPre['MONTHS']
dfPre['GoodBorrower'] = np.where(filter1 & filter2 & filter3 & filter4, 1, 0)
#percentage "good borrowers"
print("Percent Accepted:")
print(len(dfPre[dfPre['GoodBorrower'] == 1])/length1)


#LabelEncoded Dataframe
dfEnc = dfPre.drop(columns=['0', '1', '2', '3', '4', '5', 'X', 'C', 'MONTHS', 'MONTHS_BALANCE', 'PercentOnTime', 'PercentNoLoan', 'Percent0to1', 'Percent1to2', 'PercentOver2', 'OverTwoMonths', 'PaidorNoLoan'])
cat_var = dfEnc.select_dtypes(include='object').columns.to_list()
for var in cat_var:
    dfEnc[var] = label_encoder.fit_transform(dfEnc[var])


#make variables more interpretable 
dfEnc.isna().sum()
dfEnc['Age'] = (round(-dfEnc['DAYS_BIRTH'] / 365, 0)).astype(int)
dfEnc = dfEnc.drop(['DAYS_BIRTH'], axis=1)

dfEnc['Employed_period'] = (round(-dfEnc['DAYS_EMPLOYED']/365, 0)).astype(int)
dfEnc['Employed_period'] = dfEnc['Employed_period'].apply(lambda x: 0 if x<0 else x) # Set negative to 0 which means no unemployed
dfEnc = dfEnc.drop(['DAYS_EMPLOYED'], axis=1)
continuous_var =  ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Employed_period', 'Age']


"""Variable transformation """
#plot hist for continuous var to determine normality
plt.subplots_adjust(hspace=1)
plt.suptitle("Continuous Vars ")
for i, col in enumerate(continuous_var):
    ax = plt.subplot(2, 2, i + 1)
    sns.distplot((dfEnc[col])).set_title(f' {col}')

#plot log transform
plt.subplots_adjust(hspace=1)
plt.suptitle("log1p(Continuous Var) ")
for i, col in enumerate(continuous_var):
    ax = plt.subplot(2, 2, i + 1)
    sns.distplot(np.log1p(dfEnc[col])).set_title(f' {col}')

#log transform to improve normality
log_cols = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Employed_period']

for col in log_cols:
    dfEnc[col] = dfEnc[col].apply(lambda x: np.log1p(x))

#split into x and y
X = dfEnc.drop(columns=['ID','GoodBorrower'])
Y = dfEnc['GoodBorrower']
#scale data
scaler = StandardScaler(with_std=True,
                        with_mean=True)
X_sc = scaler.fit_transform(X)
#train test split
(X_train, X_test ,Y_train ,Y_test) = \
                                    skm.train_test_split(X_sc,
                                    Y, test_size=0.2, random_state=0)
#smote                                    
X_train_sm,Y_train_sm = SMOTE().fit_resample(X_train,Y_train)

"""LDA"""

lda = LDA(store_covariance=True)
lda.fit(X_train_sm, Y_train_sm)
y_pred_lda =lda.predict(X_test)
#cross val
K=5
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)
result_lda = cross_validate(lda, X_train_sm, Y_train_sm , cv=kfold)
result_lda['test_score'].mean()
cm_lda = confusion_matrix(Y_test, y_pred_lda)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm_lda, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
print('Accuracy is {:.5}'.format(accuracy_score(Y_test, y_pred_lda)))
print('F1 Score is {:.5}'.format(f1_score(Y_test, y_pred_lda)))
print('Precission Score is {:.5}'.format(precision_score(Y_test, y_pred_lda)))


"""QDA"""

qda = QDA(store_covariance=True)              

result_QDA = cross_validate(qda, X_train_sm, Y_train_sm , cv=kfold)
result_QDA['test_score'].mean()
qda.fit(X_train_sm, Y_train_sm)
y_pred_qda =qda.predict(X_test)

cm_qda = confusion_matrix(Y_test, y_pred_qda)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm_qda, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix QDA')

print('Accuracy is {:.5}'.format(accuracy_score(Y_test, y_pred_qda)))
print('F1 Score is {:.5}'.format(f1_score(Y_test, y_pred_qda)))
print('Precission Score is {:.5}'.format(precision_score(Y_test, y_pred_qda)))

"""Naive Bayes"""

NB = GaussianNB()
result_NB = cross_validate(NB, X_train_sm, Y_train_sm , cv=kfold)
result_QDA['test_score'].mean()
NB.fit(X_train_sm, Y_train_sm)
y_pred_NB =NB.predict(X_test)

cm_NB = confusion_matrix(Y_test, y_pred_NB)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm_NB, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix NB')

print('Accuracy is {:.5}'.format(accuracy_score(Y_test, y_pred_NB)))
print('F1 Score is {:.5}'.format(f1_score(Y_test, y_pred_NB)))
print('Precission Score is {:.5}'.format(precision_score(Y_test, y_pred_NB)))

"""log reg"""
lr_gen = LogisticRegression(class_weight= 'balanced')
result_lr = cross_validate(NB, X_train_sm, Y_train_sm , cv=kfold)
result_lr['test_score'].mean()
lr_gen.fit(X_train_sm, Y_train_sm)

y_pred_lr =lr_gen.predict(X_test)


cm_lr = confusion_matrix(Y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix LR')

print('Accuracy is {:.5}'.format(accuracy_score(Y_test, y_pred_lr)))
print('F1 Score is {:.5}'.format(f1_score(Y_test, y_pred_lr)))
print('Precission Score is {:.5}'.format(precision_score(Y_test, y_pred_lr)))
