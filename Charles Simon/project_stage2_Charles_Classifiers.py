#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:46:31 2023

@author: charlessimon
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

import numpy as np
import seaborn as sns
import math
from sklearn import preprocessing

#for log reg
from sklearn.linear_model import LogisticRegression

#for PCA
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA

#for grid search and cv
from sklearn.pipeline import Pipeline
import sklearn.model_selection as skm
#scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#column trans
from sklearn.compose import ColumnTransformer
# metrics and cross val
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_score

#Smote
from imblearn.over_sampling import SMOTE

#operating system tools
import os

#decision tree
from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR,
                          plot_tree ,
                          export_text)

# set path 
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
#boxplots
dfPre.boxplot(column = ['Percent0to1', 'Percent1to2', 'PercentOver2', 'PercentNoLoan'])
plt.show()
dfPre.boxplot(column = ['PaidorNoLoan', 'PercentOnTime'])
plt.show()

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



#heatmap to look at variable relationships
sns.heatmap(dfEnc.corr())
plt.show()

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



"""PCA Analysis"""


#scale data
scaler = StandardScaler(with_std=True,
                        with_mean=True)

X_sc_pca = scaler.fit_transform(X)

#inital PCA fitting
pca = PCA()
pca.fit(X_sc_pca)
pca.mean_
pca.components_


#Biplot
font = {
    "family": "serif",
    "color": "black",
    "weight": "normal",
    "size": 8,
}
i, j = 0, 1
scores = pca.transform(X_sc_pca)
scale_arrow = s_ = 8
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:,0], scores[:,1])
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
ax.set_ylim(-5,5)
ax.set_xlim(-5,8)

for k in range(10):#first 10 comp
    ax.arrow(0, 0, s_*pca.components_[i,k], s_*pca.components_[j,k])
    ax.text(s_*pca.components_[i,k],
            s_*pca.components_[j,k],
            X.columns[k], fontdict = font)
ax.set_position([1,1, 1, 1])

#explaind varianc
scores.std(0, ddof=1)
pca.explained_variance_
pca.explained_variance_ratio_

#elbow plot
fig, axes = plt.subplots(2, 1, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ticks = np.arange(pca.n_components_)+1
ax = axes[0]
ax.plot(ticks,
        pca.explained_variance_ratio_,
        marker='o')
ax.set_xlabel('Principal Component', fontsize = 15);
ax.set_ylabel('Proportion of Variance Explained',fontsize = 15)
ax.set_ylim([0,0.2])
ax.set_xlim([1,17])
ax.set_xticks(ticks)
ax = axes[1]
ax.plot(ticks,
        pca.explained_variance_ratio_.cumsum(),
        marker='o')
ax.set_xlabel('Principal Component', fontsize = 15)
ax.set_ylabel('Cumulative Proportion of Variance Explained', fontsize = 15)
ax.set_ylim([0.15, 1])
ax.set_xlim([1,17])
ax.set_xticks(ticks)
fig

#find optimal number of components for log reg
pcaLR = PCA()
LR = LogisticRegression(class_weight = 'balanced')
pipe = Pipeline([('scaler', scaler),
                 ('pca', pcaLR),
                 ('logreg', LR)])
#5 fold cv
K=5
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)

X.shape
param_grid = {'pca__n_components': range(1, 16)}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=kfold,
                        scoring='accuracy', verbose=10)
grid.fit(X, Y)
#plot mean MSE vs num comp
pcr_fig, ax = subplots(figsize=(8,8))
n_comp = param_grid['pca__n_components']
ax.errorbar(n_comp,
grid.cv_results_['mean_test_score'],
grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Accuracy', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])

#retrive optimal components
print('optimal number of components is' ,grid.best_params_.get('pca__n_components'))
grid.best_params_.get('pca__n_components')
print('Best acc' ,grid.best_score_)

#get score for 1 component
pca3= PCA(n_components= 6)
LR3_comp = LogisticRegression(class_weight = 'balanced')
X_pca_6 =pca3.fit_transform(X_sc_pca)
crossval_acc_6 = cross_val_score(LR3_comp, X_pca_6, Y, cv=kfold).mean()
print('accuracy of cross valadation with optimal number of components is',crossval_acc_6 )
y_pred = skm.cross_val_predict(LR3_comp, X_pca_6, Y, cv=5)
cm = confusion_matrix(Y, y_pred)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()




"""Lasso and general logre"""
#general log reg
Kfold_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
lr_gen = LogisticRegression(class_weight= 'balanced')
result = cross_validate(lr_gen, X_sc_pca, Y, cv=Kfold_strat)
result['test_score'].mean()
y_pred_gen = skm.cross_val_predict(lr_gen, X_sc_pca, Y, cv=Kfold_strat)
cm = confusion_matrix(Y, y_pred_gen)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#lasso log reg

#coef regularization plot
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                 test_size=0.3,
                 random_state=0,
                 stratify=Y)


X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
X_std = scaler.fit_transform(X)
weights, params = [], [] 
for c in np.arange(-4., 4.):# linear vector from -10 to 10
    lr_lasso = LogisticRegression(penalty='l1',class_weight= 'balanced', C=10.**c,
                        solver='liblinear',  random_state=0)
    lr_lasso.fit(X_train_std, y_train)
    weights.append(lr_lasso.coef_[0])
    params.append(10**c)

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan','magenta', 'yellow', 'black','pink', 'lightgreen', 'lightblue',
'gray', 'indigo', 'orange', 'purple', 'crimson', 'chocolate', 'yellowgreen']
weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=X.columns[column ], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper right')
ax.legend(loc='upper center',
bbox_to_anchor=(1.38, 1.03),
ncol=1, fancybox=True)
plt.show()

#grid search lasso
param_grid_lasso={"C":np.logspace(-4,4,8)}
lr_lasso_hyp = LogisticRegression(penalty='l1',class_weight= 'balanced',
                     solver='liblinear',  random_state=0)
clf = skm.GridSearchCV(lr_lasso_hyp, param_grid_lasso, cv=kfold, scoring='accuracy', verbose = 10)
clf.fit(X_std, Y)
print(clf.best_params_)
a = clf.best_score_
a
#plot acc vs C
np.log(param_grid_lasso['C'])
lasso_fig, ax = subplots(figsize=(8,8))
n_comp = np.log10(param_grid_lasso['C'])
ax.errorbar(n_comp,
            clf.cv_results_['mean_test_score'],
            clf.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Accuracy', fontsize=20)
ax.set_xlabel('log(reg str)', fontsize=20)
ax.set_xticks(n_comp[::1])

lr_lasso_tuned = LogisticRegression(penalty='l1',C= clf.best_params_['C'],class_weight= 'balanced',
                     solver='liblinear',  random_state=0)

result = cross_validate(lr_lasso_tuned, X_std, Y, cv=Kfold_strat)
result['test_score'].mean()

y_pred_tuned = skm.cross_val_predict(lr_lasso_tuned, X_std, Y, cv=Kfold_strat)

cm = confusion_matrix(Y, y_pred_tuned)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()




"""Desision tree classifier"""
X_sm,Y_sm = SMOTE().fit_resample(X,Y)
#scaler line 157
X_sc_dt = scaler.fit_transform(X_sm)

dt_class = DTC(criterion='entropy', random_state=0, max_depth=15)
#kfold line 232 
result_dt = cross_validate(dt_class, X_sc_dt, Y_sm, cv=kfold)
result_dt['test_score'].mean()
X_sm.columns
y_pred_dt = skm.cross_val_predict(dt_class, X_sc_dt, Y_sm, cv=kfold)
ax = subplots(figsize=(12,12))[1]


cm = confusion_matrix(Y_sm, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
(8655+7519)/(20505+21641+8655+7519)
print('Accuracy is {:.5}'.format(accuracy_score(Y_sm, y_pred_dt)))
print('F1 Score is {:.5}'.format(f1_score(Y_sm, y_pred_dt)))
print('Precission Score is {:.5}'.format(precision_score(Y_sm, y_pred_dt)))
print('Recall Score is {:.5}'.format(recall_score(Y_sm, y_pred_dt)))


#tree pruning
(X_sm_train, X_sm_test ,Y_sm_train ,Y_sm_test) = \
                                    skm.train_test_split(X_sc_dt,
                                    Y_sm , test_size=0.5, random_state=0)
                           
dt_class.fit(X_sm_train, Y_sm_train)
ax = subplots(figsize=(12,12))[1]
plot_tree(dt_class,
          feature_names=X_sm.columns, ax=ax)
                                 
clf_dt = DTC(criterion='entropy', random_state=0) 
clf_dt.fit(X_sm_train, Y_sm_train)
accuracy_score(Y_sm_test, clf_dt.predict(X_sm_test)) 


ccp_path = clf_dt.cost_complexity_pruning_path(X_sm_train, Y_sm_train)

grid = skm.GridSearchCV(clf_dt,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True, 
                        cv=kfold,
                        scoring='accuracy', verbose=10)
grid.fit(X_sm_train, Y_sm_train)
grid.best_score_           
best_ = grid.best_estimator_                              

best_.tree_.n_leaves
best_.get_depth()
cm_grid = confusion_matrix(Y_sm_test, best_.predict(X_sm_test))
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm_grid, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

print('Accuracy is {:.5}'.format(accuracy_score(Y_sm_test, best_.predict(X_sm_test))))
print('F1 Score is {:.5}'.format(f1_score(Y_sm, y_pred_dt)))
print('Precission Score is {:.5}'.format(precision_score(Y_sm, y_pred_dt)))
print('Recall Score is {:.5}'.format(recall_score(Y_sm, y_pred_dt)))
ax = subplots(figsize=(12,12))[1]
plot_tree(best_,
          feature_names=X_sm.columns, ax=ax);

X_sm.columns



"""decision tree pca"""
X_sm,Y_sm = SMOTE().fit_resample(X,Y)
#scaler line 157
dt_class = DTC(criterion='entropy', random_state=0)

pcaDT = PCA()
pipe = Pipeline([('scaler', scaler),
                 ('pca', pcaDT),
                 ('DT', dt_class)])
#5 fold cv
K=5
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)

X.shape
param_grid = {'pca__n_components': range(1, 16)}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=kfold,
                        scoring='accuracy', verbose=10)
grid.fit(X_sm, Y_sm)
#plot mean MSE vs num comp
pcr_fig, ax = subplots(figsize=(8,8))
n_comp = param_grid['pca__n_components']
ax.errorbar(n_comp,
grid.cv_results_['mean_test_score'],
grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Accuracy', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])
print('optimal number of components is' ,grid.best_params_.get('pca__n_components'))
grid.best_params_.get('pca__n_components')
#optimal components is 14 but not much higher than 9
print('Best acc' ,grid.best_score_)
best_ = grid.best_estimator_                              

best_.tree_.n_leaves
best_._final_estimator.get_depth()

pca_opt_dt= PCA(n_components= 9)
X_pca_opt_dt =pca_opt_dt.fit_transform(X_sm_test)
best_ = grid.best_estimator_  
dt_class.fit(X_sm_train, Y_sm_train)

"""LDA"""
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
lda = LDA(store_covariance=True)
lda.fit(X_sm_train, Y_sm_train)
y_pred_lda =lda.predict(X_sm_test)

cm_lda = confusion_matrix(Y_sm_test, y_pred_lda)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm_lda, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
print('Accuracy is {:.5}'.format(accuracy_score(Y_sm_test, y_pred_lda)))
print('F1 Score is {:.5}'.format(f1_score(Y_sm_test, y_pred_lda)))
print('Precission Score is {:.5}'.format(precision_score(Y_sm_test, y_pred_lda)))
print('Recall Score is {:.5}'.format(recall_score(Y_sm_test, y_pred_lda)))


"""QDA"""

qda = QDA(store_covariance=True)
qda.fit(X_sm_train, Y_sm_train)
y_pred_qda =qda.predict(X_sm_test)

cm_qda = confusion_matrix(Y_sm_test, y_pred_qda)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm_qda, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
print('Accuracy is {:.5}'.format(accuracy_score(Y_sm_test, y_pred_qda)))

"""Naive Bayes"""
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()

NB.fit(X_sm_train, Y_sm_train)
y_pred_NB =NB.predict(X_sm_test)

cm_NB = confusion_matrix(Y_sm_test, y_pred_NB)
print('Accuracy is {:.5}'.format(accuracy_score(Y_sm_test, y_pred_NB)))
