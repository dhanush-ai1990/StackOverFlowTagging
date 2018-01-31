import re
import time
import numpy as np
#from sklearn.externals import joblib
import glob,os
import os
import time
from gensim.models import Doc2Vec
from sklearn.externals import joblib
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

import seaborn as sns

class Documents(object):
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for i, doc in enumerate(self.documents):
            yield TaggedDocument(words = doc, tags = [i])  


#code_loc='/home/dhanushd/scratch/text25/'
code_loc='/Users/Dhanush/Desktop/SODATAFINAL/CodeText25/'
name_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']

encode_label={}
i=1
for lang in name_file:
	encode_label[lang] =i
	i+=1


X=[]
y=[]
print ("Loading data")
a=time.time()

for item in name_file:
	print (item)
	code_loc_current=code_loc+item+'/'
	file_list = glob.glob(os.path.join(code_loc_current, "*.txt"))
	i = 1
	for file_path in file_list:
		f=open(file_path,'r')
		data=f.read()
		label=item
		data=data.split(" ")
		if len(data) < 3:
			continue
		X.append(data)
		y.append(encode_label[label])

print ("Data loaded")
print ("time taken to load is: " + str(time.time()-a))

print (len(X))
print (len(y))

documents = Documents(X)

#Train the Doc2Vec Model
model = Doc2Vec(size=300, dbow_words= 1, dm=0, iter=1,  window=5, seed=1337, min_count=10, workers=4,alpha=0.025, min_alpha=0.025)
model.build_vocab(documents)
for epoch in range(15):
    print("epoch "+str(epoch))
    model.train(documents, total_examples=len(X), epochs=1)
    #model.save('/home/dhanushd/scratch/Doc2Vec/Doc2Vec_text25.model')
    model.save('./Doc2Vec/Doc2Vec_codetext25.model')
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

# Generate the Document Vectors

X_vector=[]
i=0
for data in X:
	X_vector.append(model.infer_vector(data))

dumped=[X_vector,y,encode_label]
#joblib.dump(dumped,"/home/dhanushd/scratch/Doc2Vec/Doc2Vec_text25_vector.pkl",protocol=2)
joblib.dump(dumped,"./Doc2Vec/Doc2Vec_codetext25_vector.pkl",protocol=2)


