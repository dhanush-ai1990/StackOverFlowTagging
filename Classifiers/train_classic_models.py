#This program will Train 
"""
MultiNomial Naive Bayes
Random Forest
XGBoost
SVM
clustering K Means
"""
import sqlite3
import re
import time
import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#MultiNomial

def MyMultiNomialNB(X_train, y_train):
	clf = MultinomialNB()
	param_grid = {'alpha': [0.000000001,0.00000001,0.0000001,0.01,0.1,1,10,100,1000] }
	#param_grid = {'alpha': [0.000000001] }
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=5 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyExtraTreeClassifier(X_train, y_train):
	clf = ExtraTreesClassifier(min_samples_split=2, random_state=0,max_depth = 10)
	param_grid = {'n_estimators': [10,20,30,50]}
	#param_grid = {'max_depth': [1,5,10,25,50,75,100,500,1000,2000]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid,n_jobs=3)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_


def MyRandomForest(X_train, y_train):
	clf = RandomForestClassifier()
	#param_grid = {'n_estimators': [10,20,30,50,70,100]}
	param_grid = {'n_estimators': [100]}
	classifier= GridSearchCV(estimator=clf, cv=2 ,param_grid=param_grid,verbose=10)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def rbf_svm(X_train, y_train):
	rbf_svc = svm.SVC(kernel='rbf')#,max_iter = 10000,cache_size =1024,decision_function_shape ='ovo'/'ovo'
	param_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
	classifier= GridSearchCV(estimator=rbf_svc, cv=5 ,param_grid=param_grid)
	y_train= np.array(y_train)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_
"""
#print ("Running the code for Random Forest full features")
print ("Running the code for Random Forest min_df=10")
#print ("Running the code for Random Forest min_df=20")
db_file='SODATA.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
SQL = "Select  dp,tag from nouns;"
cursor.execute(SQL)

X=[]
y=[]

for row in cursor:
	X.append(row[0])
	y.append(row[1])
#Complete dataset loaded

#Change Y to categorical labels.
labels= list(set(y))
labels.sort()

label_mapping ={}

for i in range(150):
	label_mapping[labels[i]] = i

#Lets encode the training data with this labels

for i in range(len(y)):
	y[i] = label_mapping[y[i]]

a=time.time()
print ("Vectorization started")
cv = TfidfVectorizer(input ='X',stop_words = {'english'},lowercase=True,analyzer ='word',min_df=10,max_features =5000)#,non_negative=True)#,)
X = cv.fit_transform(X)
vocab = np.array(cv.get_feature_names())
print (len(vocab))
print ("Time taken to vectorize is %s seconds" %(time.time()-a))#print vocab

"""
#Lets split the data into train-test
[X,y]=joblib.load('/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Database/DocVectors/Text25VectorP2.pkl')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=453456)

X_train = X_train[0:100000]
y_train  =y_train[0:100000]
results_ = MyRandomForest(X_train,y_train)
print (results_['mean_train_score'])
print (results_['mean_test_score'])
"""
clf = MultinomialNB(alpha=1)

clf.fit(X_train,y_train)
y_pred  = clf.predict(X_test)

print precision_recall_fscore_support(y_test, y_pred, average='weighted')
print f1_score(y_test, y_pred, average='weighted')
print(accuracy_score(y_test, y_pred))
"""






