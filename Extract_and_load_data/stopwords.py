from spacy.en.language_data import STOP_WORDS
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import spacy
from sklearn.externals import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math

en_stopwords = stopwords.words('english')

stopwords= set(en_stopwords+list(STOP_WORDS)+list(ENGLISH_STOP_WORDS))
stopwords= [w.decode('utf-8')for w in stopwords]


nlp = spacy.load('en')

doc = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")

for token in doc:
   print(token, token.lemma, token.lemma_)

raise Exception("stop")

from sklearn.externals import joblib

tag_dict=joblib.load("/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Pickles/tagcount.pkl")

i=1
count =0
y =[]
x=[]
for key, value in sorted(tag_dict.iteritems(), key=lambda (k,v): (v,k),reverse=True):
	count+=value 
	#print "%s: %s" % (key, value)
	print key
	y.append(value)
	x.append(i)

	if i >150:
		break
	i+=1

print count
sns.set()
sns.set_style("dark")
sns.set_style("darkgrid")
sns.set_context("poster")
title=plt.title('Plot showing number of questions posted for top 500 SO tags')
plt.setp(title, color='black')
plt.ylabel('Number of posts')
plt.xlabel('Tags')
plt.xlim(1,500)
plt.bar(x,y,align='center',color='green')

#plt.show()
#plt.savefig("/users/Dhanush/Desktop/TagHistogram.png",dpi=500)


