from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.externals import joblib
from matplotlib import pyplot
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
import string
import seaborn as sns
X=[]
y=[]
"""
for row in cursor:
	X.append(row[0])
	y.append(row[1])
"""

code_loc='/Users/Dhanush/Desktop/SODATAFINAL/code25/'
name_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']
name_file=['groovy']
for item in name_file:
	code_loc_current=code_loc+item+'/'
	file_list = glob.glob(os.path.join(code_loc_current, "*.txt"))
	for file_path in file_list:
		f=open(file_path,'r')
		data=f.read().translate(None, string.punctuation).split(" ")
		data=[it.lower() for it in data if len(it) > 2]
		label=item
		X.append(data)
		y.append(label)
	#Complete dataset loaded
#print(X)
#Change Y to categorical labels.
"""
labels= list(set(y))
labels.sort()
print len(labels)
label_mapping ={}

for i in range(24):
	print i
	label_mapping[labels[i]] = i+1

print(label_mapping)
#Lets encode the training data with this labels

#for i in range(len(y)):
#	y[i] = label_mapping[y[i]]

"""

model = Word2Vec(X, min_count=500, size=300, window=3,workers=3)

tsne = TSNE(n_components=2, init='random', random_state=0)

model.save('text25_word2vec.model')

X_wv = model[model.wv.vocab]




result = tsne.fit_transform(X_wv)

print ("size of vocab is below")
print (len(model.wv.vocab))

"""

print model.most_similar('oil')
print model.similarity('fraud', 'ferc')


"""
plt.figure(figsize=(15, 5))
sns.set_style("whitegrid",{"xtick.major.size": 6})
sns.set(font_scale=1.4)

pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.tight_layout()
pyplot.gcf().subplots_adjust(bottom=0.15,left=0.05)
sns.plt.title('Plot Showing Code Feature Representation for Java in 2D Space',weight='bold').set_fontsize('16')
sns.plt.ylabel('Dimension 2',weight='bold').set_fontsize('16')
sns.plt.xlabel('Dimension 1',weight='bold').set_fontsize('16')
#plt.xlim(-600,125)
pyplot.show()

#plt.savefig("/Users/Dhanush/Desktop/ObjCCode.png", dpi=300)


