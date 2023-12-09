# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:46:45 2023

@author: charlessimon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing
import sklearn.model_selection as skm


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


from scipy.stats import shapiro, chi2_contingency, fisher_exact
from scipy import stats


from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from yellowbrick.classifier import ClassificationReport
#Crossval
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve



#moddels
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

label_encoder = preprocessing.LabelEncoder()

"""create classification"""
#get dataframe from separate csv's
df1 = pd.read_csv('application_record.csv', header = 0)
df2 = pd.read_csv('credit_record.csv', header = 0)
#creating dataframe merged
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
"""classification created"""

"""drop colums used to determine worthiness"""
dfEnc = dfPre.drop(columns=['0', '1', '2', '3', '4', '5', 'X', 'C', 'MONTHS', 'MONTHS_BALANCE', 'PercentOnTime', 'PercentNoLoan', 'Percent0to1', 'Percent1to2', 'PercentOver2', 'OverTwoMonths', 'PaidorNoLoan'])
#describe current data frame
a =dfEnc.describe()
#Flag mobile has only one value and will be useless
dfEnc = dfEnc.drop(['FLAG_MOBIL'], axis =1)
#categorical var misses some
cat_var = dfEnc.select_dtypes(include='object').columns.to_list()


"""Chi-square test"""
num_cat = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'] #categorical variables represented as 1s and 0s in og data
cat_var.extend(num_cat)
df_chi_fish_test = pd.DataFrame()
df_chi_fish_test['feature'] = cat_var
vals = []
for var in cat_var:
    a = pd.crosstab(dfEnc[var],dfEnc['GoodBorrower'] )
    if (a.shape[0] == 2) & (a.shape[1] == 2):
        vals.append(fisher_exact(a, alternative='two-sided')[1])
    else:
        vals.append(chi2_contingency(a)[1])
df_chi_fish_test['pvalue'] = vals
X5_labels = df_chi_fish_test[df_chi_fish_test['pvalue']<0.05].feature.tolist() #alpha =0.05 level significant variables
X10_labels = df_chi_fish_test[df_chi_fish_test['pvalue']<0.10].feature.tolist() #alpha =0.10 level significant variables


"""label encode and clean"""
#label encode
cat_var = dfEnc.select_dtypes(include='object').columns.to_list()
for var in cat_var:
    dfEnc[var] = label_encoder.fit_transform(dfEnc[var])

#make variables more interpretable 
dfEnc['Age'] = (round(-dfEnc['DAYS_BIRTH'] / 365, 0)).astype(int) #age in years
dfEnc = dfEnc.drop(['DAYS_BIRTH'], axis=1) #drop days birth

dfEnc['Employed_period'] = (round(-dfEnc['DAYS_EMPLOYED']/365, 0)).astype(int) #employed period in years
dfEnc['Employed_period'] = dfEnc['Employed_period'].apply(lambda x: 0 if x<0 else x) # Set negative to 0 which means no unemployed
dfEnc = dfEnc.drop(['DAYS_EMPLOYED'], axis=1)
continuous_var =  ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Employed_period', 'Age', 'CNT_CHILDREN']
dfEnc.isna().sum()


"""log transform"""
log_cols = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Employed_period', 'CNT_CHILDREN']

for col in log_cols:
    dfEnc[col] = dfEnc[col].apply(lambda x: np.log1p(x))

"""T-test"""
df_t_test_log = pd.DataFrame()
df_t_test_log['feature'] = continuous_var
df_good = dfEnc[dfEnc['GoodBorrower']==1]
df_bad = dfEnc[dfEnc['GoodBorrower']==0]

vals = []
for var in continuous_var:
    if stats.levene(df_good[var], df_bad[var]).pvalue>0.05:
        vals.append(stats.ttest_ind(df_good[var], df_bad[var]).pvalue)
    else:
        vals.append(stats.ttest_ind(df_good[var], df_bad[var], equal_var=False).pvalue)
df_t_test_log['t-test_p'] = vals    
t_test_feature10and5 =df_t_test_log[df_t_test_log['t-test_p']<0.05].feature.tolist()

"""Features that are significant at alpha = 0.05 and 0.10 levels"""
X5_labels= X5_labels+t_test_feature10and5 #alpha = 0,05
X10_labels = X10_labels + t_test_feature10and5 #alpha = 0.10
#random forest showed age and income important
X5_with_extras = X5_labels + ['Age', 'AMT_INCOME_TOTAL']
"""split into x and y"""
X = dfEnc.drop(columns=['ID','GoodBorrower'])
X5 =X[X5_labels] #X for variables significant at 0.05 alpha level
X10 = X[X10_labels] #X for variables significant at 0.10 alpha level
Y = dfEnc['GoodBorrower'] #classification labels

"""scale, train test split, smote"""
#Full dataset
scaler = StandardScaler(with_std=True,
                        with_mean=True)
X_sc = scaler.fit_transform(X)

(X_train, X_test ,Y_train ,Y_test) = \
                                    skm.train_test_split(X_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X_sm,Y_sm = SMOTE().fit_resample(X_train,Y_train)

#Reduced 0.05 alpha
X5_sc = scaler.fit_transform(X5)

(X5_train, X5_test ,Y5_train ,Y5_test) = \
                                    skm.train_test_split(X_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X5_sm,Y5_sm = SMOTE().fit_resample(X5_train,Y_train)

#reduced 0.10 alpha
X5_sc = scaler.fit_transform(X5)

(X5_train, X5_test ,Y5_train ,Y5_test) = \
                                    skm.train_test_split(X_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X5_sm,Y5_sm = SMOTE().fit_resample(X5_train,Y_train)

#kfold
K=5
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)

# reduced alpha 0.05 with age and annual income
"""Classifcation evaluation functions"""
"""Confusion matrix heat map"""
#input Y_test and Y predict by classifier
def cm(test, pred):
    conf = confusion_matrix(test, pred)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    print('Balance Accuracy is {:.5}'.format(balanced_accuracy_score(test, pred)))
    print('Accuracy is {:.5}'.format(accuracy_score(test, pred)))
    print('F1 Score is {:.5}'.format(f1_score(test, pred)))
    print('Precission Score is {:.5}'.format(precision_score(test, pred)))
    print('Recall Score is {:.5}'.format(recall_score(test, pred)))
"""clasification report heat map"""
#input training X and Y, test x and y and the classifer mod
def class_rep(X_train,Y_train, X_Test, Y_Test, clf):
    visualizer = ClassificationReport(clf, support=True)
    visualizer.fit(X_train, Y_train)        # Fit the visualizer and the model
    visualizer.score(X_Test, Y_Test)
    visualizer.show()

"""For all models"""
"""check smote vs class weight on full if available"""
"""Fit full data, then X5"""
"for best perfomer gridsearch"
"""LDA (only SMOTE)"""
#Full
lda = LDA(store_covariance=True)
lda.fit(X_sm, Y_sm)
y_pred_lda =lda.predict(X_test)
cm(Y_test, y_pred_lda)
class_rep(X_sm, Y_sm, X_test,Y_test, lda)


#Reduced X5
lda = LDA(store_covariance=True)
lda.fit(X5_sm, Y5_sm)
y_pred_lda =lda.predict(X5_test)
cm(Y5_test, y_pred_lda)
class_rep(X5_sm, Y5_sm, X5_test,Y5_test, lda)

#GridsearchCV both comprable use X5
model = Pipeline([
        ('sampling', SMOTE()),
        ('clf', LDA(store_covariance=True, solver='lsqr'))
    ])
param_grid = dict(clf__shrinkage=np.linspace(0,1,num=10 ))
grid = GridSearchCV(model, param_grid, cv=kfold, scoring="balanced_accuracy", return_train_score=False, verbose = 10)
grid.fit(X5_sc,Y)

grid.best_score_
best = grid.best_estimator_    
crossval_lda = cross_val_score(best, X5_sc, Y, cv=kfold).mean()
print(crossval_lda)
crossval_lda = cross_val_score(best, X5_sc, Y, cv=kfold, scoring='balanced_accuracy').mean()
print(crossval_lda)
"""QDA(only smote"""
#Full
qda = QDA()
qda.fit(X_sm, Y_sm)
y_pred_qda =qda.predict(X_test)
cm(Y_test, y_pred_qda)
class_rep(X_sm, Y_sm, X_test,Y_test, qda)

#reduced
qda = QDA()
qda.fit(X5_sm, Y5_sm)
y_pred_qda =qda.predict(X5_test)
cm(Y5_test, y_pred_qda)
class_rep(X5_sm, Y5_sm, X5_test,Y5_test, qda)

#gridshearch
model = Pipeline([
        ('sampling', SMOTE()),
        ('clf', QDA())
    ])
param_grid = dict(clf__reg_param=np.linspace(0,1,num=10 ))
grid = GridSearchCV(model, param_grid, cv=kfold, scoring="balanced_accuracy", return_train_score=False, verbose = 10)
grid.fit(X5_sc,Y)

grid.best_score_
best = grid.best_estimator_    
crossval_qda = cross_val_score(best, X5_sc, Y, cv=kfold).mean()
print(crossval_qda)
crossval_qda = cross_val_score(best, X5_sc, Y, cv=kfold, scoring='balanced_accuracy').mean()
print(crossval_qda)

"""Logistic Reg(class weights and SMOTE)"""
#smote full
lr = LogisticRegression()
lr.fit(X_sm, Y_sm)
y_pred_lr =lr.predict(X_test)
cm(Y_test, y_pred_lr)
class_rep(X_sm, Y_sm, X_test,Y_test, lr)

#class weights
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, Y_train)
y_pred_lr =lr.predict(X_test)
cm(Y_test, y_pred_lr)
class_rep(X_train, Y_train, X_test,Y_test, lr)


#class weights performed better for most other models
#no diff here so use class weight
lr = LogisticRegression(class_weight='balanced')
lr.fit(X5_train, Y5_train)
y_pred_lr =lr.predict(X5_test)
cm(Y5_test, y_pred_lr)
class_rep(X5_train, Y5_train, X5_test,Y5_test, lr)

#gridsearchcv
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet'],
    'C' : np.logspace(-4, 4, 20),
    }
]

grid = GridSearchCV(lr, param_grid = param_grid, cv = kfold, scoring='balanced_accuracy', verbose=10, n_jobs=-1)
grid.fit(X5_sc,Y)
grid.best_score_

best =grid.best_estimator_

crossval_lf = cross_val_score(best, X5_sc, Y, cv=kfold).mean()
print(crossval_qda)
crossval_lf = cross_val_score(best, X5_sc, Y, cv=kfold, scoring='balanced_accuracy').mean()
print(crossval_qda)

"""Naive Bayes(smote only"""
#full
NB = GaussianNB()
NB.fit(X_sm, Y_sm)
y_pred_NB =NB.predict(X_test)
cm(Y_test, y_pred_NB)
class_rep(X_sm, Y_sm, X_test,Y_test, NB)
#reduced
NB = GaussianNB()
NB.fit(X5_sm, Y5_sm)
y_pred_NB =NB.predict(X5_test)
cm(Y5_test, y_pred_NB)
class_rep(X5_sm, Y5_sm, X5_test,Y5_test, NB)
