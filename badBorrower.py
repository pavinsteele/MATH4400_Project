#a brief look into what would happen if any default indicated a bad borrower
#results did not seem better from very quick look

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
#from yellowbrick.classifier import ClassificationReport
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


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
filter1 = (dfPre['Percent0to1']) == 0
#People who have no more than five percent due by over 2 months (column 1)
filter2 = (dfPre['Percent1to2']) == 0
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

print("We only accept 12 percent of people")
print("Let's look at a neural network for this result")

df = dfEnc
X = df.drop(columns= ['GoodBorrower', 'ID']).values
Y = df['GoodBorrower'].values
Xscaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xscaled, Y,
random_state=1, test_size=0.25, stratify=Y)
print(X_train.shape)
print(y_train.shape)
#Performing SMOTE
X_train,y_train = SMOTE().fit_resample(X_train,y_train)
print(X_train.shape)
print(y_train.shape)

clf = MLPClassifier(random_state=1, max_iter=300, activation="relu")
clf.fit(X_train, y_train)
clf.predict_proba(X_test[:1])
clf.predict(X_test[:5, :])
print(X_test.shape)
print(y_test.shape)
clf.score(X_test, y_test)

predict_train = clf.predict(X_train)
predict_test = clf.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train,labels=clf.classes_))
print(classification_report(y_train,predict_train))

print(confusion_matrix(y_test,predict_test))
cm = confusion_matrix(y_test,predict_test)
print(classification_report(y_test,predict_test))

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['BadBorrower', 'GoodBorrower']); ax.yaxis.set_ticklabels(['BadBorrower', 'GoodBorrower']);
plt.show()