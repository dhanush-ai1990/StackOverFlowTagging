import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from time import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
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
from scipy.stats import mode
import seaborn as sns

[X,y,encode_label]=joblib.load('/Users/Dhanush/Desktop/SODATAFINAL/Doc2Vec/Doc2Vec_text25_vector.pkl')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=453456)
"""
X_train=X_train[1:500]
y_train=y_train[1:500]

reduced_data_rpca = PCA(n_components=2).fit_transform(X_train)
#kmeans = KMeans(init='k-means++', n_clusters=24, n_init=10)
#kmeans.fit(reduced_data_rpca)

target_names=encode_label.keys()

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray','green','cyan',]
#colors = sns.color_palette('gist_rainbow', n_colors=24)
j=1
y_train=np.array(y_train)
#colors=['black', 'blue', 'purple', 'yellow',]
for i in range(len(colors)):

	x = reduced_data_rpca[:, 0][y_train == j]
	y = reduced_data_rpca[:, 1][y_train == j]
	plt.scatter(x, y, c=colors[i])
	j+=1
#plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.savefig("soem.png", dpi=300)


"""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=24)
kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_train)
y_train=np.array(y_train)
labels = np.zeros_like(y_kmeans)
for i in range(1,25,1):
    mask = (y_kmeans == i)
    print (mask)
    labels[mask] = mode(y_train[mask])[0]
X_train=np.array(X_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_kmeans, s=50, cmap='viridis')
from sklearn.metrics import accuracy_score
print accuracy_score(y_train, labels)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.savefig("soem.png", dpi=300)
