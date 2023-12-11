import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing
import sklearn.model_selection as skm


from imblearn.over_sampling import SMOTE

from scipy.stats import shapiro, chi2_contingency, fisher_exact
from scipy import stats


from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from yellowbrick.classifier import ClassificationReport

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


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


'''#percentage "good borrowers"
print("Percent Accepted:")
print(len(dfPre[dfPre['GoodBorrower'] == 1])/length1)'''
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
'''num_cat = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'] #categorical variables represented as 1s and 0s in og data
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
X10_labels = df_chi_fish_test[df_chi_fish_test['pvalue']<0.10].feature.tolist() #alpha =0.10 level significant variables'''


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
'''log_cols = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Employed_period', 'CNT_CHILDREN']

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
X10_labels = X10_labels + t_test_feature10and5 #alpha = 0.10'''

"""split into x and y"""
#X = dfEnc.drop(columns=['ID','GoodBorrower'])
features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'Age', 'Employed_period']
#features = ['FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS']
X = dfEnc.loc[:, features]
#X5 =X[X5_labels] #X for variables significant at 0.05 alpha level
#X10 = X[X10_labels] #X for variables significant at 0.10 alpha level
Y = dfEnc['GoodBorrower'] #classification labels

"""scale, train test split, smote"""
scaler = StandardScaler(with_std=True,
                        with_mean=True)
X_sc = scaler.fit_transform(X)

(X_train, X_test ,Y_train ,Y_test) = \
                                    skm.train_test_split(X_sc,
                                    Y, stratify=Y, test_size=0.2, random_state=0)
#X_sm,Y_sm = SMOTE(random_state=0).fit_resample(X_train,Y_train)
"""Classifcation evaluation functions"""
"""Confusion matrix heat map"""
#input Y_test and Y predict by classifier
'''def cm(test, pred):
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
    visualizer.show()'''

# Logistic Regression

print('-Features used in model: ', features)
print('')

model = LogisticRegression(class_weight='balanced')

model.fit(X_train, Y_train)
Y_Predict = model.predict(X_test)

confMatrix = confusion_matrix(Y_test, Y_Predict)

print(f'-Accuracy of Model = {str(accuracy_score(Y_test, Y_Predict))}')
print('-Confusion Matrix:')
print(confMatrix)
print('')

# RFE

print('Logistic Regression After RFE:')
print('')
num_features = 3
rfe = RFE(model, n_features_to_select= num_features)
rfe = rfe.fit(X_train, Y_train)

features = np.array(features)
new_features = features[rfe.support_]
print('-Top', num_features, 'features from RFE algorithm: ', new_features)
print('')

X = dfEnc.loc[:, new_features]
# Y stays the same

X_sc = scaler.fit_transform(X)

(X_train, X_test ,Y_train ,Y_test) = skm.train_test_split(X_sc, Y, stratify=Y, test_size=0.2, random_state=0)
#X_sm, Y_sm = SMOTE(random_state=0).fit_resample(X_train, Y_train)

model.fit(X_train, Y_train)

Y_Predict = model.predict(X_test)
confMatrix = confusion_matrix(Y_test, Y_Predict)

print('-Features used in model: ', new_features)
print('')
print(f'-Accuracy of Model = {str(accuracy_score(Y_test, Y_Predict))}')
print('-Confusion Matrix:')
print(confMatrix)
print("\n")
