from xgboost.sklearn import XGBClassifier
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
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st

#print ("Running the code for XGBoost min_df=20")
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
cv = TfidfVectorizer(input ='X',stop_words = {'english'},lowercase=True,analyzer ='word',min_df=20)#,non_negative=True)#,max_features =75000)
X = cv.fit_transform(X)
vocab = np.array(cv.get_feature_names())
print (len(vocab))
print ("Time taken to vectorize is %s seconds" %(time.time()-a))#print vocab


#Lets split the data into train-test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=453456)
print (X_train.shape)





params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}
clf = XGBClassifier()
print ("Fitting")
clf.fit(X_train, y_train)

#classifier= GridSearchCV(estimator=algo,param_grid=param_grid)
y_pred  = clf.predict(X_test)

print (precision_recall_fscore_support(y_test, y_pred, average='weighted'))
print (f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))



