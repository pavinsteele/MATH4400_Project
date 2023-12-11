#initial preprocessing file
#replaced by Charlie's preprocessing.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn import preprocessing

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
dfPre = dfPre.groupby('ID').agg({'MONTHS_BALAN-1E': 'min', '0': 'sum', '1': 'sum', '2': 'sum', '3': 'sum', '4': 'sum', '5': 'sum', 'C': 'sum', 'X' : 'sum'})
#merge

dfPre['MONTHS'] = (dfPre['MONTHS_BALAN-1E'] - 1) * -1

dfPre = pd.merge(df1, dfPre, on = 'ID')

#index: ID
#bool (Y/N or 1/0) variables: FLAG_OWN_CAR (Y/N), FLAG_OWN_REALTY (Y/N), FLAG_MOBIL (1/0), FLAG_WORK_PHONE (1/0), FLAG_PHONE (1/0), FLAG_EMAIL (1/0)
#numerical variables: AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED, XXYEARS_EMPLOYEDXX, XXCNT_CHILDRENXX, CNT_FAM_MEMBERS
#categorical variables: CODE_GENDER, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE

dfPre = dfPre.drop(columns=['CNT_CHILDREN', 'DAYS_BIRTH', 'YEARS_EMPLOYED', 'FLAG_MOBIL'])

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


#LabelEncoded DataBase
dfEnc = dfPre.drop(columns=['0', '1', '2', '3', '4', '5', 'X', 'C', 'MONTHS', 'MONTHS_BALAN-1E', 'PercentOnTime', 'PercentNoLoan', 'Percent0to1', 'Percent1to2', 'PercentOver2', 'OverTwoMonths', 'PaidorNoLoan'])
dfEnc['NAME_INCOME_TYPE']= label_encoder.fit_transform(dfEnc['NAME_INCOME_TYPE'])
dfEnc['NAME_EDUCATION_TYPE']= label_encoder.fit_transform(dfEnc['NAME_EDUCATION_TYPE'])
dfEnc['NAME_FAMILY_STATUS']= label_encoder.fit_transform(dfEnc['NAME_FAMILY_STATUS'])
dfEnc['NAME_HOUSING_TYPE']= label_encoder.fit_transform(dfEnc['NAME_HOUSING_TYPE'])
dfEnc['OCCUPATION_TYPE']= label_encoder.fit_transform(dfEnc['OCCUPATION_TYPE'])
dfEnc['CODE_GENDER']= label_encoder.fit_transform(dfEnc['CODE_GENDER'])
dfEnc['FLAG_OWN_CAR']= label_encoder.fit_transform(dfEnc['FLAG_OWN_CAR'])
dfEnc['FLAG_OWN_REALTY']= label_encoder.fit_transform(dfEnc['FLAG_OWN_REALTY'])
dfEnc['FLAG_WORK_PHONE']= label_encoder.fit_transform(dfEnc['FLAG_WORK_PHONE'])
dfEnc['FLAG_PHONE']= label_encoder.fit_transform(dfEnc['FLAG_PHONE'])
dfEnc['FLAG_EMAIL']= label_encoder.fit_transform(dfEnc['FLAG_EMAIL'])
#Categorical Database
dfCat = dfPre


#heatmap to look at variable relationships
sns.heatmap(dfEnc.corr())
plt.show()
