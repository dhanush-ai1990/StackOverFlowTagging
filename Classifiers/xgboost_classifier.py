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
import glob,os
#MultiNomial

from xgboost import XGBClassifier

param = {
	'n_estimators':[50,100,150,200],
	'max_depth':[2,3,4,5,6,7,8,9],
	'min_child_weight':[2,3,4,5],
	'colsample_bytree':[0.2,0.6,0.8],
	'colsample_bylevel':[0.2,0.6,0.8]
}

clf=XGBClassifier(objective= 'multi:softprob')

gsearch1 = GridSearchCV(estimator = clf, param_grid = param,cv=2,verbose = 100)

print ("classifier Loaded")

"""
db_file='oldData.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
SQL = "Select  date,name from code25;"
cursor.execute(SQL)
"""
X=[]
y=[]
"""
for row in cursor:
	X.append(row[0])
	y.append(row[1])
"""
code_loc='/Users/MSR/Desktop/SODATA25/CodeText25/'
name_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']
for item in name_file:
	print item
	code_loc_current=code_loc+item+'/'
	file_list = glob.glob(os.path.join(code_loc_current, "*.txt"))
	i = 1
	for file_path in file_list:
		f=open(file_path,'r')
		data=f.read()
		label=item
		X.append(data)
		y.append(label)
	#Complete dataset loaded
#print(X)
#Change Y to categorical labels.
labels= list(set(y))
labels.sort()

label_mapping ={}

for i in range(24):
	label_mapping[labels[i]] = i

print(label_mapping)
#Lets encode the training data with this labels

for i in range(len(y)):
	y[i] = label_mapping[y[i]]
#print(y)
a=time.time()
print ("Vectorization started")
cv = TfidfVectorizer(input ='X',stop_words = {'english'},lowercase=True,analyzer ='word',min_df=10,max_features =5000)#,non_negative=True)#,)
X = cv.fit_transform(X)
vocab = np.array(cv.get_feature_names())
print (len(vocab))
print ("Time taken to vectorize is %s seconds" %(time.time()-a))#print vocab

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=453456)




        
gsearch1.fit(X_train, y_train)       
print gsearch1.bestscore
print gsearch1.bestparams    












