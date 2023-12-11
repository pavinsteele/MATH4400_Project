#this was an attempt to use TensorFlow for neuralNet
#It did not work, likely due to user error
#Will be fixed, but for now refer to neuralNet3

#help from here
#https://www.analyticsvidhya.com/blog/2021/10/implementing-artificial-neural-networkclassification-in-python-from-scratch/
#https://www.youtube.com/watch?v=E35CVhVKISA
#Importing necessary Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Smote
from imblearn.over_sampling import SMOTE

#put file into directory from Charlie's preprocessing file
#just to make code easier to read
df = pd.read_csv('dfUltimate.csv', header = 0)
print(list(df.columns))
X = df.drop(columns= ['GoodBorrower', 'ID', 'Unnamed: 0']).values
Y = df['GoodBorrower'].values
X_sm,Y_sm = SMOTE().fit_resample(X,Y)

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_sm,Y_sm,test_size=0.2,random_state=0)


#Performing Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initialising ANN
ann = tf.keras.models.Sequential()

#Adding First Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

#Adding Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

#Adding Output Layer
#sigmoid vs softplus vs softsign vs sigmoid
ann.add(tf.keras.layers.Dense(units=1,activation="softplus"))

#Compiling ANN
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Fitting ANN
ann.fit(X_train,Y_train,batch_size=5000,epochs = 100)

#Predicting result for Single Observation
#True
#print(ann.predict(sc.transform([[1,0,4,1,47.58082192,-1000.665753,0,18,2,0]])) > 0.5)
#False
print(ann.predict(sc.transform([[1,1,1,1,46.22465753,2.106849315,1,0,2,2]])) > 0.5)

#X_test is 3d, ann.predict expects 2d
y_true = Y_test
y_pred = ann.predict(X_test)
cm = tf.math.confusion_matrix(labels= y_true, predictions = y_pred)
cm = cm.numpy()
print("_______________________________________________________________")
#print(cm)
#total1=sum(sum(cm))
#accuracy1=(cm[0,0]+cm[1,1])/total1
#print ('Accuracy : ', accuracy1)



ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['GoodBorrower', 'BadBorrower']); ax.yaxis.set_ticklabels(['BadBorrower', 'GoodBorrower']);
plt.show()


#getting about 53% for training accuracy and 50% for testing. I'm guessing this has to do with
#the composition of pos vs neg in the pieces of data rather than actual accuracy
#all testing negative or positive