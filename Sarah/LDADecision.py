#This program has LDA, Decision Tree, and KMeans algorithms
#Some with SMOTE and some without

from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.simplefilter("ignore")


import warnings
# import data and drop, so only ten variables
#put file into directory from Charlie's preprocessing file
#just to make code easier to read
df = pd.read_csv('dfUltimate.csv', header = 0)

scale = StandardScaler()


#LDA
#some help from here https://www.youtube.com/watch?v=czVaCse00VM
print("LDA w/o SMOTE: ")
model = LDA()
X = df.drop(columns = ['GoodBorrower'])
y = df['GoodBorrower']

counter = Counter(y)
print(counter)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
X_train = scale.fit_transform(X_train)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
X_test = model.transform(X_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy = {str(accuracy_score(y_test, y_pred))}')
total1=sum(sum(cm))
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)



#smote
#help from here https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
print("____________________________________________________")
print("LDA with Smote: ")

...
# transform the dataset
oversample = SMOTE()

counter = Counter(y)
print(counter)
X = scale.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

X_train_sm, y_train_sm = oversample.fit_resample(X_train, y_train)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy = {str(accuracy_score(y_test, y_pred))}')
total1=sum(sum(cm))
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)



#This but QDA
print("____________________________________________________")
print("QDA with SMOTE: ")
model = QDA()

model.fit(X_train, y_train)

counter = Counter(y)
print(counter)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy = {str(accuracy_score(y_test, y_pred))}')
total1=sum(sum(cm))
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)



# Decision Tree w/ SMOTE
print("____________________________________________________")
print('Decision Tree w SMOTE: ')

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')  # 'gini' is the default
decision_tree = decision_tree.fit(X_train_sm, y_train_sm)
y_pred = decision_tree.predict(X_test)
# print(decision_tree.feature_importances_)


counter = Counter(y)
print(counter)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)



# Decision Tree w/o SMOTE
print("____________________________________________________")
print("Decision Tree w/o SMOTE: ")
X = df.drop(columns = ['GoodBorrower'])
y = df['GoodBorrower']

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')  # 'gini' is the default
decision_tree = decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
# print(decision_tree.feature_importances_)


counter = Counter(y)
print(counter)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# K Mean Clustering
print("____________________________________________________")
print("K Means Clustering")
inertias = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,10), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters = 4, random_state = 0, n_init='auto')
kmeans.fit(X_train)

print(kmeans.predict(X_test))
print(kmeans.labels_)
print(y_test)