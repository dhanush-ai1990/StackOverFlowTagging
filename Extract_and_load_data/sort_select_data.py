from sklearn.externals import joblib

postid_with_tags=joblib.load("/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Pickles/tag_ids.pkl")


tags = postid_with_tags.keys()
print len(tags)
list_tags =[]
for tag in tags:
	if tag =='ssl':
		continue
	ids=postid_with_tags[tag]
	ids.sort(reverse=True)
	if len(ids) <5000:
		print tag
	list_tags.extend(ids[0:5000])


print len(list_tags)

joblib.dump(list_tags,"/Users/Dhanush/Desktop/Projects/StackOverFlowTagging/Pickles/tags5000_ids.pkl")