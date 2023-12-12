import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

#read in csv file using pandas package
# how to merge: https://saturncloud.io/blog/how-to-merge-two-csv-files-into-one-with-pandas-by-id/

#label encoder object
label_encoder = preprocessing.LabelEncoder()

#get databases from separate csv's
df1 = pd.read_csv('application_record.csv', header = 0)
df2 = pd.read_csv('credit_record.csv', header = 0)


#creating database merged
dfPre = pd.get_dummies(df2['STATUS'], drop_first = False)
dfPre = pd.concat([df2, dfPre], axis=1)
dfPre = dfPre.groupby('ID').agg({'MONTHS_BALANCE': 'min', '0': 'sum', '1': 'sum', '2': 'sum', '3': 'sum', '4': 'sum', '5': 'sum', 'C': 'sum', 'X' : 'sum'})
#merge

dfPre['MONTHS'] = (dfPre['MONTHS_BALANCE'] - 1) * -1

dfPre = pd.merge(df1, dfPre, on = 'ID')

#index: ID
#bool (Y/N or 1/0) variables: FLAG_OWN_CAR (Y/N), FLAG_OWN_REALTY (Y/N), FLAG_MOBIL (1/0), FLAG_WORK_PHONE (1/0), FLAG_PHONE (1/0), FLAG_EMAIL (1/0)
#numerical variables: AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED, XXYEARS_EMPLOYEDXX, XXCNT_CHILDRENXX, CNT_FAM_MEMBERS
#categorical variables: CODE_GENDER, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE

#dfPre = dfPre.drop(columns=['CNT_CHILDREN', 'DAYS_BIRTH', 'YEARS_EMPLOYED', 'FLAG_MOBIL'])
#dfPre = dfPre.drop(columns=['CNT_CHILDREN', 'DAYS_BIRTH', 'FLAG_MOBIL']) #YEARS_EMPLOYED not in csv

#testing diff columns dropped
dfPre = dfPre.drop(columns=['FLAG_MOBIL', 'FLAG_EMAIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE'])

#finding our Acceptance Column
dfPre['PaidorNoLoan'] = (dfPre['C'] + dfPre['X'])/dfPre['MONTHS']
dfPre['OverTwoMonths'] = dfPre['2'] + dfPre['3'] + dfPre['4'] + dfPre['5']
#percentages
dfPre['PercentNoLoan'] = dfPre['X']/dfPre['MONTHS']
dfPre['PercentOnTime'] = dfPre['C']/dfPre['MONTHS']
dfPre['Percent0to1'] = dfPre['0']/dfPre['MONTHS']
dfPre['Percent1to2'] = dfPre['1']/dfPre['MONTHS']
dfPre['PercentOver2'] = dfPre['OverTwoMonths']/dfPre['MONTHS']
'''#boxplots
dfPre.boxplot(column = ['Percent0to1', 'Percent1to2', 'PercentOver2', 'PercentNoLoan'])
plt.show()
dfPre.boxplot(column = ['PaidorNoLoan', 'PercentOnTime'])
plt.show()'''

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
dfPre['GoodBorrower'] = np.where(filter1 & filter2 & filter3 & filter4, 'Y', 'N')

'''#percentage "good borrowers"
print("Percent Accepted:")
print(len(dfPre[dfPre['GoodBorrower'] == 1])/length1)'''


#LabelEncoded DataBase
dfEnc = dfPre.drop(columns=['0', '1', '2', '3', '4', '5', 'X', 'C', 'MONTHS', 'MONTHS_BALANCE', 'PercentOnTime', 'PercentNoLoan', 'Percent0to1', 'Percent1to2', 'PercentOver2', 'OverTwoMonths', 'PaidorNoLoan'])
dfCat = dfPre.drop(columns=['0', '1', '2', '3', '4', '5', 'X', 'C', 'MONTHS', 'MONTHS_BALANCE', 'PercentOnTime', 'PercentNoLoan', 'Percent0to1', 'Percent1to2', 'PercentOver2', 'OverTwoMonths', 'PaidorNoLoan'])
dfEnc['NAME_INCOME_TYPE']= label_encoder.fit_transform(dfEnc['NAME_INCOME_TYPE'])
dfEnc['NAME_EDUCATION_TYPE']= label_encoder.fit_transform(dfEnc['NAME_EDUCATION_TYPE'])
dfEnc['NAME_FAMILY_STATUS']= label_encoder.fit_transform(dfEnc['NAME_FAMILY_STATUS'])
dfEnc['NAME_HOUSING_TYPE']= label_encoder.fit_transform(dfEnc['NAME_HOUSING_TYPE'])
dfEnc['OCCUPATION_TYPE']= label_encoder.fit_transform(dfEnc['OCCUPATION_TYPE'])
dfEnc['CODE_GENDER']= label_encoder.fit_transform(dfEnc['CODE_GENDER'])
dfEnc['FLAG_OWN_CAR']= label_encoder.fit_transform(dfEnc['FLAG_OWN_CAR'])
dfEnc['FLAG_OWN_REALTY']= label_encoder.fit_transform(dfEnc['FLAG_OWN_REALTY'])
dfEnc['GoodBorrower']= label_encoder.fit_transform(dfEnc['GoodBorrower'])
#dfEnc['FLAG_WORK_PHONE']= label_encoder.fit_transform(dfEnc['FLAG_WORK_PHONE'])
#dfEnc['FLAG_PHONE']= label_encoder.fit_transform(dfEnc['FLAG_PHONE'])
#dfEnc['FLAG_EMAIL']= label_encoder.fit_transform(dfEnc['FLAG_EMAIL'])
#Categorical Database
#dfCat

# Logistic Regression

logregDFEnc = dfEnc.copy()

logregDFEnc = logregDFEnc.drop("ID", axis='columns')  # not important based on corr

#features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS']
features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS']  # removed DAYS_EMPLOYED unreliable?

X = logregDFEnc.loc[:, features]
y = logregDFEnc['GoodBorrower']

print('-Features used in model: ', features)
print('')

model = LogisticRegression(class_weight='balanced')
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

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
rfe = RFE(model, n_features_to_select=8)
rfe = rfe.fit(X, y)

features = np.array(features)
new_features = features[rfe.support_]
#print('-Top three features from RFE algorithm: ', new_features)
#print('')

X = logregDFEnc.loc[:, new_features]
# Y stays the same

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, Y_train)
Y_Predict = model.predict(X_test)
confMatrix = confusion_matrix(Y_test, Y_Predict)

print('-Features used in model: ', new_features)
print('')
print(f'-Accuracy of Model = {str(accuracy_score(Y_test, Y_Predict))}')
print('-Confusion Matrix:')
print(confMatrix)
print("\n")
exit()
# Visualizations for top 8 features

visDF = dfCat[['DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'CNT_FAM_MEMBERS', 'NAME_FAMILY_STATUS', 'FLAG_OWN_REALTY', 'GoodBorrower']].copy()

visDF['DAYS_BIRTH'] = visDF['DAYS_BIRTH'].abs()
visDF['DAYS_BIRTH'] = visDF['DAYS_BIRTH']/365
visDF['DAYS_BIRTH'] = (visDF['DAYS_BIRTH'].round()).astype(int)

GoodDF = visDF.copy()
GoodDF = GoodDF.drop(GoodDF[GoodDF['GoodBorrower'] == 'N'].index)

BadDF = visDF.copy()
BadDF = BadDF.drop(BadDF[BadDF['GoodBorrower'] == 'Y'].index)

# DAYS_BIRTH

'''print(visDF['DAYS_BIRTH'].describe())
print(GoodDF['DAYS_BIRTH'].describe())
print(BadDF['DAYS_BIRTH'].describe())'''

plt.hist(GoodDF['DAYS_BIRTH'], color='cyan', edgecolor='black')
plt.hist(BadDF['DAYS_BIRTH'], color='maroon', edgecolor='black')

plt.title("Count of Good and Bad Borrowers for Age")

plt.legend(['Good Borrower', 'Bad Borrower'])

plt.show()

# AMT_INCOME_TOTAL

'''print(visDF['AMT_INCOME_TOTAL'].describe())
print(GoodDF['AMT_INCOME_TOTAL'].describe())
print(BadDF['AMT_INCOME_TOTAL'].describe())'''


plt.hist(GoodDF['AMT_INCOME_TOTAL'], color='cyan', edgecolor='black')
plt.hist(BadDF['AMT_INCOME_TOTAL'], color='maroon', edgecolor='black')

plt.title("Count of Good and Bad Borrowers for Total Income")

plt.legend(['Good Borrower', 'Bad Borrower'])

plt.show()

# OCCUPATION_TYPE
pd.crosstab(visDF['OCCUPATION_TYPE'], visDF['GoodBorrower']).plot(kind="bar", stacked=True, rot=90, fontsize=5, figsize=(5, 8), color=['maroon', 'cyan'], edgecolor='black')

plt.show()

# NAME_EDUCATION_TYPE
pd.crosstab(visDF['NAME_EDUCATION_TYPE'], visDF['GoodBorrower']).plot(kind="bar", stacked=True, rot=0, fontsize=5, figsize=(6, 4), color=['maroon', 'cyan'], edgecolor='black')

plt.show()

# NAME_INCOME_TYPE

pd.crosstab(visDF['NAME_INCOME_TYPE'], visDF['GoodBorrower']).plot(kind="bar", stacked=True, rot=0, fontsize=5, figsize=(4, 4), color=['maroon', 'cyan'], edgecolor='black')

plt.show()

# CNT_FAM_MEMBERS
GoodDF['CNT_FAM_MEMBERS'] = GoodDF['CNT_FAM_MEMBERS'].astype(int)
BadDF['CNT_FAM_MEMBERS'] = BadDF['CNT_FAM_MEMBERS'].astype(int)

plt.hist(GoodDF['CNT_FAM_MEMBERS'], color='cyan', edgecolor='black')
plt.hist(BadDF['CNT_FAM_MEMBERS'], color='maroon', edgecolor='black')

plt.title("Count of Good and Bad Borrowers for each Family Size")

plt.legend(['Good Borrower', 'Bad Borrower'])

plt.show()

# NAME_FAMILY_STATUS

pd.crosstab(visDF['NAME_FAMILY_STATUS'], visDF['GoodBorrower']).plot(kind="bar", stacked=True, rot=0, fontsize=5, figsize=(4, 4), color=['maroon', 'cyan'], edgecolor='black')

plt.show()

# FLAG_OWN_REALTY

pd.crosstab(visDF['FLAG_OWN_REALTY'], visDF['GoodBorrower']).plot(kind="bar", stacked=True, rot=0, fontsize=5, figsize=(4, 4), color=['maroon', 'cyan'], edgecolor='black')

plt.show()

# Logistic Regression plot

features = ['FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'DAYS_BIRTH', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS']  # removed DAYS_EMPLOYED unreliable?

X = logregDFEnc.loc[:, features]
y = logregDFEnc['GoodBorrower']
