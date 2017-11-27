import json, sys, os, xmltodict, csv
from os.path import join
from utils import *
import shutil
from sklearn.externals import joblib
import time

import sys
reload(sys)
sys.setdefaultencoding('utf8')

output ='/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Database/posts.csv'
tag_file = open('/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Textfiles/tags.txt','r')
postid_with_tags=joblib.load("/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Pickles/tags5000_ids.pkl")
def clean(x):
    #neo4j-import doesn't support: multiline (coming soon), quotes next to each other and escape quotes with '\""'
    return x.replace('\n','').replace('\r','').replace('\\','').replace('"','')


def open_csv(name):
    return csv.writer(open(output, 'w'), doublequote=False, escapechar='\\')



#Read the tag list file for SO and load into dictionary for faster hashing.
tag_file=['bash','c++','php','javascript','sql','c#','html','c','r','python','css','perl','objective-c','java','vb.net','ruby','swift','haskell','lua','scala']
print len(tag_file)

tag_dict ={}
for tag in tag_file:
    #tag=tag.split('\n')[0]
    tag_dict[tag] = 0
"""
tag_dict ={}
for item in postid_with_tags:
    tag_dict[item]=0
"""
posts = open_csv('posts')
posts.writerow(['serial','postid', 'title', 'body','tag','score','creationdate'])
count =1
j=1

a = time.time()
for i, line in enumerate(open("/Volumes/HDD/Projects/StackOverFlowData/Posts.xml")):
    line = line.strip()
    if i %100000==0:
        print (i,time.time() -a)
        a = time.time()
    try:
        if line.startswith("<row"):
            el = xmltodict.parse(line)['row']
            el = replace_keys(el)
            #print(el)
            postid= el.get('id')
            posttype = el['posttypeid']
            

            if (int(posttype) ==1): #These are questions, we need only title,Body and Tag(Primary)
                Score =0
                CreationDate=el['creationdate'][0:10].replace('-',"")
                Score=clean(el.get('score',''))
                j+=1
                if el.get('tags'):
                    eltags = [x.replace('<','') for x in el.get('tags').split('>')]
                    tags= [x.lower() for x in eltags if x]
                    tag=tags[0]
                    if tag in tag_dict:
                        tag_dict[tag]= tag_dict[tag] +1 
                        postid=el['id']
                        title=clean(el.get('title',''))
                        body=clean(el.get('body',''))
                        posts.writerow([count,postid,title,body,tag,Score,CreationDate])
                        count+=1
            else:
                continue
    except Exception as e:
        print('x',e)


print(i,'posts ok')
joblib.dump(tag_dict,"/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Pickles/tag20_count.pkl")

print ("Total Posts")
print (i)

print ("Questions total")
print (j)

print ("Questions selected")
print (count)

