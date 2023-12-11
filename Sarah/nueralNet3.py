from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


df = pd.read_csv('dfUltimate.csv', header = 0)
print(list(df.columns))
X = df.drop(columns= ['GoodBorrower', 'ID', 'Unnamed: 0']).values
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