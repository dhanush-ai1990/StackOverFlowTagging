import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr

from sklearn.externals import joblib


char_dict=joblib.load("dict_file_char.pkl")
loc_dict=joblib.load("dict_file_loc.pkl")
labels=list(loc_dict.keys())


mean =[]
median=[]
cor_loc =[]
cor_char=[]
pearson_lang=[]
for label in labels:
	mean.append(np.mean(char_dict[label]))
	cor_loc.extend(loc_dict[label])
	median.append(np.median(char_dict[label]))
	cor_char.extend(char_dict[label])
	pearson_lang.append(pearsonr(loc_dict[label], char_dict[label])[0])

x_coordinates = np.arange(len(mean))
#print (mean)
#print (median)
"""
plt.figure(figsize=(16, 12))
plt.subplot(1, 2, 1)

sns.set()
sns.set_style("dark",{"xtick.major.size": 4,"ytick.major.size": 4})
x_coordinates = np.arange(len(mean))
plt.bar(x_coordinates, mean, align='center',color='#A93226')
plt.xticks(x_coordinates,labels)
plt.xticks(rotation=90)
plt.title("Mean- char of code for SO Programming languages")
plt.ylabel('Mean ')

plt.subplot(1, 2, 2)

x_coordinates = np.arange(len(median))
sns.set_style("dark",{"xtick.major.size": 4,"ytick.major.size": 4})
plt.bar(x_coordinates, median, align='center',color='#A93226')
plt.xticks(x_coordinates,labels)
plt.xticks(rotation=90)
plt.title("Mean- char of code for SO Programming languages")
plt.ylabel('Mean ')
#plt.savefig("mean_median_char.png", dpi=300)


print (len(cor_loc))

print (len(cor_char))

cor_loc=np.array(cor_loc)
cor_char=np.array(cor_char)
"""
print (pearsonr(cor_loc, cor_char)[0])
plt.figure(figsize=(10, 10))
sns.set()
sns.set_style("white",{"xtick.major.size": 4,"ytick.major.size": 4})
plt.scatter(x_coordinates,pearson_lang,color='#A93226', s=25, marker="o")
plt.xticks(x_coordinates,labels)
plt.xticks(rotation=90)
plt.title("Scatter plot showing correlation of LOC with # of char")
plt.ylabel('Pearson Correlation ')
plt.savefig("Scatter_loc_char_pearson.png", dpi=300)
