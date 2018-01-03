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
from sklearn.metrics import classification_report
from scipy.stats.stats import pearsonr
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
    plt.ylabel('True label')
    plt.xlabel('predicted Label')




def GetMetrics():
	name_file= ['c++','bash','php','javascript','sql','c#','html','c','r','python','css','perl','objective-c','java','vb.net','ruby','swift','haskell','lua','scala']
	loc_lang_loc={}
	loc_lang_pred={}
	labeL_mapping ={}
	for i in range(len(name_file)):
		labeL_mapping[name_file[i]]=i
		loc_lang_pred[name_file[i]]=[]
		loc_lang_loc[name_file[i]]=[]
	#print (labeL_mapping)
	y_pred =[]
	y_test =[]
	code_loc_current='/Users/Dhanush/Downloads/codeout'
	file_list = glob.glob(os.path.join(code_loc_current, "*.txt"))



	for file_path in file_list:
		algofile=open(file_path,'r')
		out=algofile.readline().replace("\n","").split(" ")
		size=out[3]
		loc=out[1]
		#if size=='small':
		name=out[0]
		loc_lang_loc[out[0]].append(int(loc))
		name=labeL_mapping[name]
		y_test.append(name)
		predicted=algofile.readline().replace("\n","").split(" ")[0]
		if predicted=='markdown':
			predicted=algofile.readline().replace("\n","").split(" ")[0]
		if predicted=='vb':
			predicted='vb.net'
		

		predicted=labeL_mapping[predicted]
		if (name==predicted):
			loc_lang_pred[out[0]].append(1)
		else:
			loc_lang_pred[out[0]].append(0)
		y_pred.append(predicted)

	print (precision_recall_fscore_support(y_test, y_pred, average='weighted'))
	print (f1_score(y_test, y_pred, average='weighted'))
	print(accuracy_score(y_test, y_pred))
	#cnf_matrix = confusion_matrix(y_test, y_pred)
	#sns.set()
	#sns.set_style("dark")
	#sns.set_style("white",{"xtick.major.size": 5})
	#plt.figure(figsize=(14,12))
	#print classification_report(y_test, y_pred, target_names=name_file)
	#plot_confusion_matrix(cnf_matrix, classes=name_file,title='Confusion matrix for Programming Languages')
	#plt.savefig("ConfusionMatrixMedium.png", dpi=300)	
	#plt.figure(figsize=(10, 10))
	"""
	x_coordinates = np.arange(20)
	sns.set()
	temp=[]
	sns.set_style("white",{"xtick.major.size": 4,"ytick.major.size": 4})
	plt.figure(figsize=(10, 10))
	for i in range(len(name_file)):
		lines=loc_lang_loc[name_file[i]]
		prede=loc_lang_pred[name_file[i]]
		temp.append(pearsonr(lines,prede)[0])
	plt.scatter(x_coordinates,temp, color='#4A235A',s=25, marker="o")
	plt.xticks(x_coordinates,name_file)
	plt.xticks(rotation=90)
	plt.title("Scatter plot showing correlation of LOC with Classification Accuracy")
	plt.ylabel('Prediction, 1=Correct,0=False ')
	#plt.show()
	plt.savefig("Scatter_Pearson_prediction_Loc.png", dpi=300)
	"""
	lines=[]
	prede=[]
	for i in range(len(name_file)):
		lines.extend(loc_lang_loc[name_file[i]])
		prede.extend(loc_lang_pred[name_file[i]])
	print pearsonr(lines,prede)[0]


GetMetrics()