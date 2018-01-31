#Train Final Classifiers and make predictions and generate confusion matrix
from xgboost import XGBClassifier
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import glob,os
from sklearn.datasets import make_blobs
import os
import glob
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
from sklearn.externals import joblib
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()


X=[]
y=[]
code_loc='/Users/Dhanush/Desktop/SODATAFINAL/CodeText25/'
#code_loc='/home/dhanushd/scratch/code25/'
name_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']
for item in name_file:
	print (item)
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

#print(label_mapping)
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



print ("Training Classifier")

clf=XGBClassifier(objective= 'multi:softprob',n_estimators=200,colsample_bytree=0.2,max_depth=5,min_child_weight=3,colsample_bylevel=0.2)
clf.fit(X_train, y_train) 
print ("Predicting Classifier")
y_pred=clf.predict(X_test)

print (precision_recall_fscore_support(y_test, y_pred, average='weighted'))
print (f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))

cnf_matrix = confusion_matrix(y_test, y_pred)
sns.set_style("whitegrid",{"xtick.major.size": 6})
sns.set(font_scale=1.1)
#plt.figure(figsize=(14,12))
plot_confusion_matrix(cnf_matrix,normalize=True, classes=name_file,title='Confusion Matrix for Code Only Classifier')

sns.plt.title('Confusion Matrix for Text Only Classifier').set_fontsize('12')
sns.plt.ylabel('True label').set_fontsize('12')
sns.plt.xlabel('Predicted Label').set_fontsize('12')

plt.savefig("/Users/Dhanush/Desktop/CMTextCodeOnly.png", dpi=200)


joblib.dump([cnf_matrix,y_pred,y_test,name_file],"/Users/Dhanush/Desktop/CodeTextonly.pkl")










