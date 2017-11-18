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
from sklearn.cross_validation import ShuffleSplit
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

db_file='/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Database/SODATA.db'
def create_connection(db_file):
    # create a database connection to the SQLite database
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

def load_data(conn):
	cursor = conn.cursor()
	SQL = "Select  dp,tag from nouns;"
	# SQL = "Select * from sm_post where serial='4';"
	cursor.execute(SQL)
	title = list()

	return cursor



conn   =create_connection(db_file)
cursor = load_data(conn)

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
cv = TfidfVectorizer(input ='X',stop_words = {'english'},lowercase=True,analyzer ='word',min_df=15)#,non_negative=True)#,max_features =75000)


X = cv.fit_transform(X).toarray()
vocab = np.array(cv.get_feature_names())
print len(vocab)
print ("Time taken to vectorize is %s seconds" %(time.time()-a))#print vocab

data=[X,y,label_mapping]

joblib.dump(data,"/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Database/VectorData.pkl")







