import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import \
     GradientBoostingRegressor as GBR
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max.columns', None)
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
#print("Percent Accepted:")
#print(len(dfPre[dfPre['GoodBorrower'] == 1])/length1)


#LabelEncoded Dataframe
dfEnc = dfPre.drop(columns=['0', '1', '2', '3', '4', '5', 'X', 'C', 'MONTHS', 'MONTHS_BALANCE', 'PercentOnTime', 'PercentNoLoan', 'Percent0to1', 'Percent1to2', 'PercentOver2', 'OverTwoMonths', 'PaidorNoLoan'])
cat_var = dfEnc.select_dtypes(include='object').columns.to_list()
for var in cat_var:
    dfEnc[var] = label_encoder.fit_transform(dfEnc[var])

y = dfEnc['GoodBorrower']
X = dfEnc.drop('GoodBorrower', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# Boosting

boost = GBR(n_estimators=500,
            learning_rate=0.001,
            max_depth=3,
            random_state=0)
boost.fit(X_train, y_train)
y_hat_boost = boost.predict(X_test)
print("MSE with Boosting using 500 estimators and a learning rate of 0.001")
print(np.mean((y_test - y_hat_boost)**2))


boost2 = GBR(n_estimators=500,
            learning_rate=0.2,
            max_depth=3,
            random_state=0)
boost2.fit(X_train, y_train)
y_hat_boost2 = boost2.predict(X_test)
print("MSE with Boosting using 500 estimators and a learning rate of 0.2")
print(np.mean((y_test - y_hat_boost2)**2))


# using SMOTE
X_train_smote, y_train_smote = SMOTE(random_state=1234).fit_resample(X_train, y_train)

# using XGBoost for Feature Selection

scale_pos_weight = y.shape[0] / y.sum() - 1

dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

param = { 'verbosity': 2,
          'objective': 'binary:logistic',
          'eval_metric': 'aucpr',
          'scale_pos_weight': scale_pos_weight,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'tree_method': 'gpu_hist',
          'eta': 0.1,
          'max_depth': 5,
          'gamma': 0,
          'min_child_weight': 1 }

bst = xgb.cv(param, dtrain, nfold = 3, num_boost_round = 1000, early_stopping_rounds = 50)

best_xgb = xgb.train(param, dtrain, num_boost_round = bst.shape[0])

fig, ax = plt.subplots(figsize = (6, 8))
xgb.plot_importance(best_xgb, ax = ax)
plt.show()

# using XGBoost for Model Training

param = {'learning_rate': 0.1,
         'verbosity': 2,
         'objective': 'binary:logistic',
         'tree_method': 'gpu_hist',
         'scale_pos_weight': scale_pos_weight,
         'n_estimators': 300}
xgb_grid = {'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0],
            'subsample': [0.8],
            'colsample_bytree': [0.8] }
xgbc = XGBClassifier(**param)
xgbc_cv = GridSearchCV(estimator = xgbc, param_grid = xgb_grid, cv = 3, scoring = 'average_precision', n_jobs = -1, verbose = 2)
xgbc_cv.fit(X_train, y_train)
print('Best parameters: ', xgbc_cv.best_params_)
print('Best score: ', xgbc_cv.best_score_)

print("Now using this data, we can fit the model with its best params.")

best_xgbc = XGBClassifier(scale_pos_weight = scale_pos_weight,
                          objective = 'binary:logistic',
                          tree_method = 'gpu_hist',
                          max_depth = 9,
                          min_child_weight = 1,
                          gamma = 0,
                          subsample = 0.8,
                          colsample_bytree = 0.8,
                          alpha = 0,
                          learning_rate = 0.01,
                          n_estimators = 2000)

best_xgbc.fit(X_train, y_train)
y_pred = best_xgbc.predict(X_train)

y_pred_proba_xgb = best_xgbc.predict_proba(X_test)[:, 1]
precision_xgb, recall_xgb, threshold_xgb = precision_recall_curve(y_test, y_pred_proba_xgb)
plt.plot(recall_xgb, precision_xgb, label = 'XGBoost (PRAUC = {:.3f})'.format(auc(recall_xgb, precision_xgb)))
plt.title('The Precison-Recall Curve of the XGBoost model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.show()
