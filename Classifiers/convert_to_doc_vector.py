from gensim.models import Doc2Vec
import os
import time
import pandas as pd
from sklearn.externals import joblib

#Convert class names to numbers
tag_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']
#tag_file=['python-3.x','python-2.7','python-3.5','python-3.4','python-2.x','python-3.6','python-3.3','python-2.6','java-8','java-7','c++11','c++03','c++98','c++14']
encode_label={}
i=1
for lang in tag_file:
	encode_label[lang] =i
	i+=1


output_dir2_csv='/home/dhanushd/scratch/text25.csv'

#Load the Doc2Vec Model for text
fname='text25doc2vec.model'
model = Doc2Vec.load(fname)

X=[]
y=[]
print ("Loading data")
a=time.time()


#Load the documents
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
			X.append(model.infer_vector(data))
			y.append(encode_label[label])
		except:
			continue

print (len(X))
print (len(y))
data_encoded=[X,y]
print ("time taken to process is: " + str(time.time()-a))

joblib.dump(data_encoded,"Text25Vectors.pkl")



