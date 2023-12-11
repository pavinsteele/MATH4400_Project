import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import subplots


from sklearn import preprocessing
import sklearn.model_selection as skm


from imblearn.over_sampling import SMOTE

from scipy.stats import shapiro, chi2_contingency, fisher_exact
from scipy import stats


from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from yellowbrick.classifier import ClassificationReport

from sklearn.tree import (DecisionTreeClassifier as DTC, plot_tree ,export_text) 
from sklearn.ensemble import \
(RandomForestClassifier as RFC, GradientBoostingClassifier as GBC)
from sklearn.model_selection import GridSearchCV

label_encoder = preprocessing.LabelEncoder()
from sklearn.model_selection import cross_val_score

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

"""split into x and y"""
X = dfEnc.drop(columns=['ID','GoodBorrower'])
X5 =X[X5_labels] #X for variables significant at 0.05 alpha level
X10 = X[X10_labels] #X for variables significant at 0.10 alpha level
Y = dfEnc['GoodBorrower'] #classification labels

"""scale, train test split, smote"""
scaler = StandardScaler(with_std=True,
                        with_mean=True)
X_sc = scaler.fit_transform(X)
X5_sc = scaler.fit_transform(X5)
X10_sc = scaler.fit_transform(X10)

(X5_train, X5_test ,Y5_train ,Y5_test) = \
                                    skm.train_test_split(X5_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X5_sm,Y5_sm = SMOTE().fit_resample(X5_train,Y5_train)

(X10_train, X10_test ,Y10_train ,Y10_test) = \
                                    skm.train_test_split(X10_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X10_sm,Y10_sm = SMOTE().fit_resample(X10_train,Y10_train)

(X_train, X_test ,Y_train ,Y_test) = \
                                    skm.train_test_split(X_sc,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
X_sm,Y_sm = SMOTE().fit_resample(X_train,Y_train)


K=5

kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)
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
    
"""Decision tree"""
#smote vs Class weight balance
#smote
clf_smote_comp = DTC(random_state=0)
clf_smote_comp.fit(X_sm,Y_sm)
Y_pred = clf_smote_comp.predict(X_test)
cm(Y_test,Y_pred)
clf_smote_comp.tree_.max_depth
class_rep(X_sm, Y_sm, X_test, Y_test, clf_smote_comp)


#class weight
clf_class_comp = DTC(random_state=0, class_weight='balanced')
clf_class_comp.fit(X_train,Y_train)
Y_pred = clf_class_comp.predict(X_test)
cm(Y_test,Y_pred)
clf_smote_comp.tree_.max_depth
class_rep(X_train, Y_train, X_test, Y_test, clf_class_comp)

#plot tree
ax = subplots(figsize=(14,14))[1]
plot_tree(clf_class_comp, max_depth=3,
feature_names=X.columns, ax=ax, fontsize=6, label='none');

#full data set
clf = DTC(criterion='gini',random_state=0, class_weight="balanced", max_depth=6)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
cm(Y_test,Y_pred)
clf.tree_.max_depth
class_rep(X_train, Y_train, X_test, Y_test, clf)
crossval_dt = cross_val_score(clf, X_sc, Y, cv=kfold).mean()
print(crossval_dt)

#reduced data set
clf = DTC(criterion='gini',random_state=0, class_weight="balanced", max_depth=6)
clf.fit(X5_train, Y5_train)
Y_pred = clf.predict(X5_test)
cm(Y5_test,Y_pred)
clf.tree_.max_depth
class_rep(X5_train, Y5_train, X5_test, Y5_test, clf)
crossval_dt = cross_val_score(clf, X5_sc, Y, cv=kfold).mean()
print(crossval_dt)

#optimize depth and split
clf = DTC(criterion ="gini", random_state=0, class_weight="balanced")

param_grid = {
    "max_depth":range(2, 40),
              "min_samples_split":range(2,5)}
grid = GridSearchCV(clf, param_grid, cv=kfold, scoring="balanced_accuracy", return_train_score=False, verbose = 10)
grid.fit(X5_train, Y5_train)

grid.best_params_
#gini and min sample split 4 work best

clf = DTC(criterion='gini', min_samples_split=4, random_state=0, class_weight="balanced")

param_grid = {
    "max_depth":range(2, 30),
              }
grid = GridSearchCV(clf, param_grid, cv=kfold, scoring="balanced_accuracy", return_train_score=False, verbose = 10)
grid.fit(X5_train, Y5_train)
grid.best_params_
#larger data set tends to pruduce bigger tree

plt.plot(range(2, 30), grid.cv_results_['mean_test_score'])
plt.xlabel('Value of depth for DT_clf')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

best = grid.best_estimator_
Y_pred = best.predict(X5_test)
cm(Y5_test,Y_pred)
class_rep(X5_train, Y5_train, X5_test, Y5_test, best)

clf_ccp = DTC(criterion='gini', random_state=0, class_weight="balanced",)
ccp_path = clf_ccp.cost_complexity_pruning_path(X5_train, Y5_train)
grid = skm.GridSearchCV(clf_ccp,
{'ccp_alpha': ccp_path.ccp_alphas},
    refit=True,
    cv=kfold,
    scoring='balanced_accuracy', verbose=10)
grid.fit(X5_train, Y5_train)

best_ccp = grid.best_estimator_
best_ccp.tree_.max_depth
grid.best_score_

Y_pred = best_ccp.predict(X5_test)
cm(Y5_test,Y_pred)
class_rep(X5_train, Y5_train, X5_test, Y5_test, best_ccp)
crossval_dt = cross_val_score(best_ccp, X5_sc, Y, cv=kfold).mean()
print(crossval_dt)
crossval_dt = cross_val_score(best_ccp, X5_sc, Y, cv=kfold, scoring='balanced_accuracy').mean()
print(crossval_dt)

clf = DTC(criterion='gini', random_state=0, class_weight="balanced", max_depth=6)
clf.fit(X5_train, Y5_train)

"""PCA reduced data set decision tree"""
#Use the output from PCA to create an elbow plot
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X5_sc)
scores = pca.transform(X5_sc)
pca.explained_variance_ratio_.cumsum()
#try with 3 components
pca3 = PCA(n_components=3)
pca3.fit(X5_sc)
scores3 = pca3.transform(X5_sc)
(X_train_pca, X_test_pca ,Y_train_pca ,Y_test_pca) = \
                                    skm.train_test_split(scores3,
                                    Y, stratify=Y, test_size=0.3, random_state=0)
ccp_path_pca = clf.cost_complexity_pruning_path(X_train_pca, Y_train_pca)


clf_ccp = DTC(criterion='gini', random_state=0, class_weight="balanced",)
ccp_path_pca = clf.cost_complexity_pruning_path(X_train_pca, Y_train_pca)
grid_pca = skm.GridSearchCV(clf_ccp,
{'ccp_alpha': ccp_path_pca.ccp_alphas},
    refit=True,
    cv=kfold,
    scoring='balanced_accuracy', verbose=10)
grid_pca.fit(scores3, Y)
best_ccp_pca = grid_pca.best_estimator_
best_ccp_pca.tree_.max_depth

Y_pred = best_ccp_pca.predict(X_test_pca)
cm(Y_test_pca,Y_pred)
class_rep(X_train_pca, Y_train_pca, X_test_pca, Y_test_pca, best_ccp_pca)
crossval_dt = cross_val_score(best_ccp_pca, scores3, Y, cv=kfold).mean()
print(crossval_dt)

ax = subplots(figsize=(14,14))[1]
plot_tree(best_ccp, max_depth=3,
feature_names=X5.columns, ax=ax, fontsize=6, label='none');


"""random forest and bagging"""
#bagging
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features= X5_sc.shape[1], class_weight='balanced', random_state=0)
rf.fit(X5_train, Y5_train)
Y_pred = rf.predict(X5_test)
cm(Y5_test,Y_pred)
class_rep(X5_train, Y5_train, X5_test, Y5_test, rf)
crossval_dt = cross_val_score(rf, X5_sc, Y, cv=kfold).mean()
print(crossval_dt)

grid = skm.GridSearchCV(rf,
                        {'n_estimators': range(50,2000,100)
                         },
                        refit=True, cv=kfold, scoring='balanced_accuracy',return_train_score=False, verbose = 10) 
grid.fit(X5_train, Y5_train)
grid.best_params_


plt.plot(range(50,2000,100), grid.cv_results_['mean_test_score'])
plt.xlabel('number of estimators')
plt.ylabel('Cross-Validated balanced accuracy')
plt.show()

best_bag = grid.best_estimator_
Y_pred = best_bag.predict(X5_test)
cm(Y5_test,Y_pred)
class_rep(X5_train, Y5_train, X5_test, Y5_test, best_bag)
crossval_dt = cross_val_score(best_bag, X5_sc, Y, cv=kfold).mean()
print(crossval_dt)
crossval_dt = cross_val_score(best_bag, X5_sc, Y, cv=kfold, scoring='balanced_accuracy').mean()
print(crossval_dt)

#full dataset bag
rf_full = RandomForestClassifier(max_features= X_sc.shape[1],n_estimators=1550, class_weight='balanced', random_state=0)
rf_full.fit(X_train, Y_train)
Y_pred = rf_full.predict(X_test)
cm(Y_test,Y_pred)
class_rep(X_train, Y_train, X_test, Y_test, rf_full)
crossval_dt = cross_val_score(rf, X5_sc, Y, cv=kfold).mean()
print(crossval_dt)

#Random forrest reduced
RF_clf = RFC(max_features = int(np.sqrt(X5_train.shape[1])), random_state = 0, class_weight='balanced', n_estimators =500)
RF_clf.fit(X5_train, Y5_train)
Y_pred = RF_clf.predict(X5_test)
cm(Y5_test,Y_pred)
class_rep(X5_train, Y5_train, X5_test, Y5_test, RF_clf)
crossval_dt = cross_val_score(RF_clf, X5_sc, Y, cv=kfold).mean()
print(crossval_dt)
crossval_dt = cross_val_score(best_bag, X5_sc, Y, cv=kfold, scoring='balanced_accuracy').mean()
print(crossval_dt)
feature_imp = pd.DataFrame( {'importance':RF_clf.feature_importances_}, index=X5.columns)
feature_imp.sort_values(by='importance', ascending=False)

#Random forrest full
RF_clf = RFC(max_features = int(np.sqrt(X_train.shape[1])), random_state = 0, class_weight='balanced', n_estimators =500)
RF_clf.fit(X_train, Y_train)
Y_pred = RF_clf.predict(X_test)
cm(Y_test,Y_pred)
class_rep(X_train, Y_train, X_test, Y_test, RF_clf)
crossval_dt = cross_val_score(RF_clf, X_sc, Y, cv=kfold).mean()
print(crossval_dt)
crossval_dt = cross_val_score(RF_clf, X_sc, Y, cv=kfold, scoring='balanced_accuracy').mean()
print(crossval_dt)

feature_imp = pd.DataFrame( {'importance':RF_clf.feature_importances_}, index=X.columns)
feature_imp.sort_values(by='importance', ascending=False)
