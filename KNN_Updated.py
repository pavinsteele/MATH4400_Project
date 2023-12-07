#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:56:54 2023

@author: charlessimon
"""

import numpy as np
import pandas as pd
import researchpy as rp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import os
from sklearn import preprocessing

import sklearn.model_selection as skm
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor 
from scipy.stats import shapiro, chi2_contingency, fisher_exact
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as skm
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from yellowbrick.classifier import ClassificationReport
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

os.chdir('/Users/charlessimon/Documents/Comp data science bach/Math 4500/Group Project')
os.getcwd()

#label encoder object
label_encoder = preprocessing.LabelEncoder()

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

"""drop colums used to determine worthiness"""
dfEnc = dfPre.drop(columns=['0', '1', '2', '3', '4', '5', 'X', 'C', 'MONTHS', 'MONTHS_BALANCE', 'PercentOnTime', 'PercentNoLoan', 'Percent0to1', 'Percent1to2', 'PercentOver2', 'OverTwoMonths', 'PaidorNoLoan'])
#describe current data frame
a =dfEnc.describe()
#Flag mobile has only one value and will be useless
dfEnc = dfEnc.drop(['FLAG_MOBIL'], axis =1)
#categorical var misses some
cat_var = dfEnc.select_dtypes(include='object').columns.to_list()

"""Chi-square test"""
num_cat = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
cat_var.extend(num_cat)
df_chi_fish_test = pd.DataFrame()
df_chi_fish_test['feature'] = cat_var

vals,vals2 = [],[]
for var in cat_var:
    a = pd.crosstab(dfEnc[var],dfEnc['GoodBorrower'] )
    b,expected = rp.crosstab(dfEnc[var],dfEnc['GoodBorrower'],expected_freqs= True )
    c = (expected>5).values.tolist()
    if (a.shape[0] == 2) & (a.shape[1] == 2):
        vals.append(fisher_exact(a, alternative='two-sided')[1])
        vals2.append('met')
    else:
        vals.append(chi2_contingency(a)[1])
        if(all(c)):
            vals2.append('met')
        else:
            vals2.append('check')
df_chi_fish_test['pvalue'] = vals
df_chi_fish_test['Assumptions'] = vals2

print(df_chi_fish_test[df_chi_fish_test['pvalue']<0.05])
print(df_chi_fish_test[df_chi_fish_test['pvalue']<0.10])
X5_labels = df_chi_fish_test[df_chi_fish_test['pvalue']<0.05].feature.tolist()
X10_labels = df_chi_fish_test[df_chi_fish_test['pvalue']<0.10].feature.tolist()
X15_labels = df_chi_fish_test[df_chi_fish_test['pvalue']<0.15].feature.tolist()
#flag_phone , flag_work_phone worst, Gender next Income type Housing_type email, own car all greater than 0,05, but not as sig
"""label encode and clean"""
#label encode
cat_var = dfEnc.select_dtypes(include='object').columns.to_list()
for var in cat_var:
    dfEnc[var] = label_encoder.fit_transform(dfEnc[var])

#make variables more interpretable 
dfEnc['Age'] = (round(-dfEnc['DAYS_BIRTH'] / 365, 0)).astype(int)
dfEnc = dfEnc.drop(['DAYS_BIRTH'], axis=1)

dfEnc['Employed_period'] = (round(-dfEnc['DAYS_EMPLOYED']/365, 0)).astype(int)
dfEnc['Employed_period'] = dfEnc['Employed_period'].apply(lambda x: 0 if x<0 else x) # Set negative to 0 which means no unemployed
dfEnc = dfEnc.drop(['DAYS_EMPLOYED'], axis=1)
continuous_var =  ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Employed_period', 'Age', 'CNT_CHILDREN']
dfEnc.isna().sum()



"""t-test continuous var"""
df_t_test = pd.DataFrame()
df_t_test['feature'] = continuous_var
df_good = dfEnc[dfEnc['GoodBorrower']==1]
df_bad = dfEnc[dfEnc['GoodBorrower']==0]

vals = []
for var in continuous_var:
    if stats.levene(df_good[var], df_bad[var]).pvalue>0.05:
        vals.append(stats.ttest_ind(df_good[var], df_bad[var]).pvalue)
    else:
        vals.append(stats.ttest_ind(df_good[var], df_bad[var], equal_var=False).pvalue)
df_t_test['t-test_p'] = vals
print(df_t_test)
#age and cnt children means don't differ gor borrower status

#boxplots for visualization
for i, col in enumerate(continuous_var):
    plt.figure(i)
    sns.boxplot(data=dfEnc, y = col, x ='GoodBorrower').set_title(f' {col}') 
    plt.show()
#

"""Shapiro Test/ normality plots"""

#Shapiro test unreliable for n>500 plot dist instead
#plot dists

plt.subplots_adjust(hspace=2,  wspace= 1)
plt.suptitle("Continuous Vars ")
for i, col in enumerate(continuous_var):
    ax = plt.subplot(3, 2, i + 1)
    sns.distplot((dfEnc[col])).set_title(f' {col}')
#Plot log dists
plt.subplots_adjust(hspace=2,  wspace= 1)
plt.suptitle("log1p(Continuous Var) ")
for i, col in enumerate(continuous_var):
    ax = plt.subplot(3, 2, i + 1)
    sns.distplot(np.log1p(dfEnc[col])).set_title(f' {col}')

#all log more normal accept age with log so transform
log_cols = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Employed_period', 'CNT_CHILDREN']

for col in log_cols:
    dfEnc[col] = dfEnc[col].apply(lambda x: np.log1p(x))


"""t-test on logged continuous variables"""
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
print(df_t_test_log)

#age and cnt children means don't differ gor borrower status

#boxplots for visualization
for i, col in enumerate(continuous_var):
    plt.figure(i)
    sns.boxplot(data=dfEnc, y = col, x ='GoodBorrower').set_title(f' {col}') 
    plt.show()
#
t_test_feature10and5 =df_t_test_log[df_t_test_log['t-test_p']<0.05].feature.tolist()
X5_labels= X5_labels+t_test_feature10and5
X10_labels = X10_labels + t_test_feature10and5
X15_labels = X15_labels + t_test_feature10and5


X = dfEnc.drop(columns=['ID','GoodBorrower'])
Y = dfEnc['GoodBorrower']
a =X.corr()
#CNT_FAMILY members and CNT_children drop 1
sns.heatmap(a, cmap="YlGnBu") 



"""variance inflation VIF"""
VifX = X
vif_data = pd.DataFrame() 
vif_data["feature"] = VifX.columns

vif_data["VIF"] = [variance_inflation_factor(VifX.values, i) 
                          for i in range(len(VifX.columns))] 
vif_data = vif_data.sort_values(by=['VIF'],ascending=False)

VifX = VifX.drop(columns=['Age', 'CNT_CHILDREN'])
vif_data = pd.DataFrame() 
vif_data["feature"] = VifX.columns

vif_data["VIF"] = [variance_inflation_factor(VifX.values, i) 
                          for i in range(len(VifX.columns))] 
vif_data = vif_data.sort_values(by=['VIF'],ascending=False)

VifX = X[X5_labels]
vif_data = pd.DataFrame() 
vif_data["feature"] = VifX.columns

vif_data["VIF"] = [variance_inflation_factor(VifX.values, i) 
                          for i in range(len(VifX.columns))] 
vif_data = vif_data.sort_values(by=['VIF'],ascending=False)

"""end vif"""
"""split/Smote and data set def"""
# removing CNT FAM members decreases overall varaince inflation substatialy
X5 =X[X5_labels]
X10 = X[X10_labels]

scaler = StandardScaler(with_std=True,
                        with_mean=True)
X_sc = scaler.fit_transform(X)
X5_sc = scaler.fit_transform(X5)
X10_sc = scaler.fit_transform(X10)

(X_train, X_test ,Y_train ,Y_test) = \
                                    skm.train_test_split(X_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X_sm,Y_sm = SMOTE().fit_resample(X_train,Y_train)

(X5_train, X5_test ,Y5_train ,Y5_test) = \
                                    skm.train_test_split(X5_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X5_sm,Y5_sm = SMOTE().fit_resample(X5_train,Y5_train)

(X10_train, X10_test ,Y10_train ,Y10_test) = \
                                    skm.train_test_split(X10_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X10_sm,Y10_sm = SMOTE().fit_resample(X10_train,Y10_train)


K=5
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)

"""Classifcation evaluation functions"""
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
def class_rep(X_train,Y_train, X_Test, Y_Test, clf):
    visualizer = ClassificationReport(clf, support=True)
    visualizer.fit(X_train, Y_train)        # Fit the visualizer and the model
    visualizer.score(X_Test, Y_Test)
    visualizer.show()
"""KNN K =1 vs K=100"""
knn1 = KNeighborsClassifier(n_neighbors=1) 
knn1.fit(X_sm, Y_sm)
knn1_pred = knn1.predict(X_test) 
cm(Y_test, knn1_pred)
class_rep(X_sm, Y_sm, X_test, Y_test, knn1)

knn100 = KNeighborsClassifier(n_neighbors=100) 
knn100.fit(X_sm, Y_sm)
knn100_pred = knn1.predict(X_test) 
cm(Y_test, knn1_pred)
class_rep(X_sm, Y_sm, X_test, Y_test, knn1)
"""ROC_AUC optimization"""
k_range = list(range(1, 50,3))


model = Pipeline([
        ('sampling', SMOTE()),
        ('clf', KNeighborsClassifier())
    ])
param_grid = dict(clf__n_neighbors=k_range)
grid = GridSearchCV(model, param_grid, cv=kfold, scoring="roc_auc", return_train_score=False, verbose = 10)
grid.fit(X_sc, Y)

plt.plot(k_range, grid.cv_results_['mean_test_score'])
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

k_range = list(range(1, 10,1))
param_grid = dict(clf__n_neighbors=k_range)
grid = GridSearchCV(model, param_grid, cv=kfold, scoring='roc_auc', return_train_score=False, verbose = 10)
grid.fit(X10_train, Y10_train)

plt.plot(k_range, grid.cv_results_['mean_test_score'])
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
grid.best_params_
#best param for full data set is 5, reduced 5 and 10 is 7
knn5 = KNeighborsClassifier(n_neighbors=5) 
knn5.fit(X_sm, Y_sm)
knn5_pred = knn5.predict(X_test) 

cm(Y_test, knn5_pred)
class_rep(X_sm, Y_sm, X_test, Y_test, knn5)

#reduced 5%
knn7 = KNeighborsClassifier(n_neighbors=7) 
knn7.fit(X5_sm, Y5_sm)
knn7_pred = knn7.predict(X5_test) 

cm(Y5_test, knn7_pred)
class_rep(X5_sm, Y5_sm, X5_test, Y5_test, knn7)

#reduced 10%
knn7.fit(X10_sm, Y10_sm)
knn7_pred = knn7.predict(X10_test) 

cm(Y10_test, knn7_pred)
class_rep(X10_sm, Y10_sm, X10_test, Y10_test, knn7)


"""Balanced accuracy comparisons"""
param_grid = dict(clf__n_neighbors=k_range)
grid = GridSearchCV(model, param_grid, cv=kfold, scoring='balanced_accuracy', return_train_score=False, verbose = 10)
grid.fit(X5_sc, Y)
grid.best_score_

plt.plot(k_range, grid.cv_results_['mean_test_score'])
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
grid.best_params_
#best param full data and 10% set 4, reduced 5%  is 6
knn4 = KNeighborsClassifier(n_neighbors=4 ) 
knn4.fit(X_sm, Y_sm)
knn4_pred = knn4.predict(X_test) 

cm(Y_test, knn4_pred)
class_rep(X_sm, Y_sm, X_test, Y_test, knn4)
modelCV = Pipeline([
        ('sampling', SMOTE()),
        ('clf', KNeighborsClassifier(n_neighbors=4))
    ])
crossval_knn_full = cross_val_score(modelCV, X_sc, Y, cv=kfold).mean()
print(crossval_knn_full)

#better accuracy for 5% AND 10% WITH K=4 BUT WORSE BALANCED ACCURACY
#reduced 5%
"""5% alpha res"""
knn6 = KNeighborsClassifier(n_neighbors=6 ) 
knn6.fit(X5_sm, Y5_sm)
knn6_pred = knn6.predict(X5_test) 

modelCV = Pipeline([
        ('sampling', SMOTE()),
        ('clf', KNeighborsClassifier(n_neighbors=6))
    ])
cm(Y5_test, knn6_pred)
class_rep(X5_sm, Y5_sm, X5_test, Y5_test, knn6)
crossval_knn_5 = cross_val_score(modelCV, X5_sc, Y, cv=kfold).mean()
print(crossval_knn_5)

#reduced 10%
modelCV = Pipeline([
        ('sampling', SMOTE()),
        ('clf', KNeighborsClassifier(n_neighbors=4))
    ])
knn4.fit(X10_sm, Y10_sm)
knn4_pred = knn4.predict(X10_test) 

cm(Y10_test, knn4_pred)
class_rep(X10_sm, Y10_sm, X10_test, Y10_test, knn4)
crossval_knn_10 = cross_val_score(modelCV, X10_sc, Y, cv=kfold).mean()
print(crossval_knn_10)


"""VISUALIZATIONS and pca"""
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
font = {
    "family": "serif",
    "color": "black",
    "weight": "normal",
    "size": 7,
}
pca = PCA(n_components=2)
pca5 = pca.fit(X5_sc)
scores= pca5.transform(X5_sc)
(X_train_pca, X_test_pca ,Y_train_pca ,Y_test_pca) = \
                                    skm.train_test_split(scores,
                                    Y, stratify=Y, test_size=0.2, random_state=0)

#grid scearch for pca data
grid = GridSearchCV(model, param_grid, cv=kfold, scoring='balanced_accuracy', return_train_score=False, verbose = 10)
grid.fit(scores, Y)
grid.best_score_

plt.plot(k_range, grid.cv_results_['mean_test_score'])
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
pca_K =grid.best_params_
modelCV_pca = Pipeline([
        ('sampling', SMOTE()),
        ('clf', KNeighborsClassifier(n_neighbors=9))])
crossval_knn_pca = cross_val_score(modelCV_pca, scores, Y, cv=kfold).mean()
print(crossval_knn_pca)

colors = ['red','green']
scale_arrow = s_ = 2
i, j = 0, 1 # which components
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:,0], scores[:,1], c = Y, cmap=matplotlib.colors.ListedColormap(colors)) 
ax.set_xlabel('PC%d' % (i+1)) 
ax.set_ylabel('PC%d' % (j+1))
for k in range(pca5.components_.shape[1]):
    ax.arrow(0, 0, s_*pca5.components_[i,k], s_*pca5.components_[j,k], color ='black') 
    ax.text(pca5.components_[i,k],
            pca5.components_[j,k], 
            X5.columns[k],
            fontdict = font)

clf = Pipeline(
    [('sampling', SMOTE()),("knn", KNeighborsClassifier(n_neighbors=8))]
)
from sklearn.inspection import DecisionBoundaryDisplay

_, axs = plt.subplots(ncols=2, figsize=(12, 5))

for ax, weights in zip(axs, ("uniform", "distance")):
    clf.set_params(knn__weights=weights).fit(X_train_pca, Y_train_pca)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_test_pca,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel='PC1',
        ylabel="PC2",
        shading="auto",
        alpha=0.5,
        ax=ax,
    )
    scatter = disp.ax_.scatter(scores[:, 0], scores[:, 1], c=Y, edgecolors="k")
    disp.ax_.legend(
        scatter.legend_elements()[0],
        Y.unique().tolist(),
        loc="best",
        title="Classes",
    )
    _ = disp.ax_.set_title(
        f"2-Class classification\n(k={clf[-1].n_neighbors}, weights={weights!r})"
    )

plt.show()
scores.columns.tolist(),

