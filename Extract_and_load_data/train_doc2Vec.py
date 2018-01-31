
import os
import time
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.externals import joblib
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

#Read the tag list file for SO and load into dictionary for faster hashing.
tag_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']
#tag_file=['python-3.x','python-2.7','python-3.5','python-3.4','python-2.x','python-3.6','python-3.3','python-2.6','java-8','java-7','c++11','c++03','c++98','c++14']

output_dir1 ='/home/dhanushd/scratch/SODATA25/code25/'
output_dir2 ='/home/dhanushd/scratch/text25/'
output_dir3 ='/home/dhanushd/scratch/CodeText25/'


output_dir1_csv ='/home/dhanushd/scratch/Code25.csv'
output_dir2_csv='/home/dhanushd/scratch/text25.csv'
output_dir3_csv='/home/dhanushd/scratch/CodeText25.csv'


class Documents(object):
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for i, doc in enumerate(self.documents):
            yield TaggedDocument(words = doc, tags = [i])  


encode_label={}
i=1
for lang in tag_file:
	encode_label[lang] =i
	i+=1


X=[]
y=[]
print ("Loading data")
a=time.time()

chunksize = 10 ** 6
for chunk in pd.read_csv(output_dir2_csv, chunksize=chunksize):
	rows = chunk.values.tolist()
	for row in rows:
		try:
			data=row[1]
			label=row[0]
			data=data.split(" ")
			if len(data) < 3:
				continue
			X.append(data)
			y.append(encode_label[label])
		except:
			continue


"""
for lang in tag_file:
	directory=output_dir2+lang+"/"
	files =os.listdir(directory)
	for f in files:
		input1=open(directory+f,'r')
		data=input1.read()
		data=data.split(" ")
		#print (data)
		X.append(data)
		y.append(encode_label[lang])
		input1.close()
"""
print ("Data loaded")
print ("time taken to load is: " + str(time.time()-a))

print (len(X))
print (len(y))

documents = Documents(X)

print ("training for code")
#iter = 1, because we keep training ourselves :)
model = Doc2Vec(size=300, dbow_words= 1, dm=0, iter=1,  window=5, seed=1337, min_count=5, workers=4,alpha=0.025, min_alpha=0.025)
model.build_vocab(documents)
for epoch in range(10):
    print("epoch "+str(epoch))
    model.train(documents, total_examples=len(X), epochs=1)
    model.save('/home/dhanushd/scratch/code25doc2vec.model')
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay







