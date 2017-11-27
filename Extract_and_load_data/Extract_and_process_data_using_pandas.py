import pandas as pd
import sqlite3
import re
from spacy.en.language_data import STOP_WORDS
from spacy.en import English
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import spacy
import time
import csv
from bs4 import BeautifulSoup
from spacy.pipeline import DependencyParser
import nltk
tag_file=['bash','c++','php','javascript','sql','c#','html','c','r','python','css','perl','objective-c','java','vb.net','ruby','swift','haskell','lua','scala']
print (len(tag_file))

tag_dict ={}
for tag in tag_file:
    #tag=tag.split('\n')[0]
    tag_dict[tag] = 0



def clean_data(text, stop_words, nlp):
	clean_text = re.sub(r'[^a-zA-Z ]', ' ', text).lower()
	# print("***************")
	# print("TEXT WITH STOP WORDS:")
	# print(clean_text)
	text_without_stop_words = remove_stopwords_lemmatize(clean_text, stop_words, nlp)
	return text_without_stop_words

def remove_stopwords_lemmatize(clean_text, stop_words, nlp):
	stop_words_removed = ' '.join(filter(lambda x: x.lower() not in stop_words,  clean_text.split()))
	doc = nlp(stop_words_removed)
	lemmatized_sentence = ' '.join([x.lemma_ for x in doc])
	return lemmatized_sentence

def open_csv(file):
	return csv.writer(open(file,'w'), doublequote=False, escapechar='\\')

def body_split(body):
	# print(body)
	soup = BeautifulSoup(body, "html5lib")
	t2_soup=soup.find_all('code')
	code_text = ""
	text_without_code = re.sub(r'<code>.*?</code>', ' ', body)
	soup2 = BeautifulSoup(text_without_code, "html5lib")
	text_without_code_tags = soup2.get_text()

	for item in t2_soup:
		code_text = code_text + str(item.text) + "\n"

	return text_without_code_tags, code_text


def NamedEntities(nlp,text):
	doc = nlp(text)
	temp =[]
	for ent in doc.ents:
		temp.append(ent.text)

	return " ".join(temp)

def dependencyparser(parser,combined):
	temp=[]
	for sentence in nltk.sent_tokenize(combined):
		parsed = parser(sentence)
		for token in parsed :
			if (token.text.isalpha()) and (len(token.text) >2):
				if (token.tag_ == "NNP") or (token.tag_ == "NNPS")or (token.tag_ == "NN"):
					temp.append(token.text)
	return " ".join(temp)




en_stopwords = stopwords.words('english')
stop_words = list(STOP_WORDS) + list(ENGLISH_STOP_WORDS) + list(en_stopwords)
nlp = spacy.load('en')
parser = English()	




title_csv = open_csv('title.csv')
title_csv.writerow(['postid','title','tag','date','score'])

body_csv = open_csv('body.csv')
body_csv.writerow(['postid','body','tag','date','score'])

code_csv = open_csv('code.csv')
code_csv.writerow(['postid','code','tag','date','score'])

dep_csv = open_csv('dependency.csv')
dep_csv.writerow(['postid','dp','tag','date','score'])

filename='/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Database/posts.csv'
chunksize = 10 ** 6

print ("Here")
start_time = time.time()
for chunk in pd.read_csv(filename, chunksize=chunksize):
	rows = chunk.values.tolist()
	for row in rows:
		score=row[5]
		date=str(row[6])
		tag=row[4]
		if tag_dict[tag] >10000:
			continue
		if int(date[0:4]) < 2012:
			continue
		if int(date[0:4]) > 2016:
			continue
		if int(score) <1:
			continue

		title = clean_data(row[2], stop_words, nlp)
		body_text, body_code = body_split(row[3])
		body  = clean_data(body_text, stop_words, nlp)
		title_csv.writerow([row[0],title,row[4]])
		body_csv.writerow([row[0],body,row[4]])
		code_csv.writerow([row[0],body_code,row[4]])
		combined = row[2] + " " +row[3]
		combined, combined_code = body_split(combined)
		dp =dependencyparser(parser,combined)
		dep=clean_data(dp, stop_words, nlp)
		tag_dict[tag] +=1
	print("--- %s seconds ---" % (time.time() - start_time))
	print (tag_dict)
	start_time = time.time()
