import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.externals import joblib
import pandas as pd
import math


'''
dict1=joblib.load("tag_20_count.pkl")
labels=list(dict1.keys())
values=list(dict1.values())
sns.set()
sns.set_style("dark",{"xtick.major.size": 4,"ytick.major.size": 4})
'''
filename='Top500Tag.csv'
chunksize = 10 ** 6
values=[]
labels=[]
for chunk in pd.read_csv(filename, chunksize=chunksize):
	rows = chunk.values.tolist()
	for row in rows:
		labels.append(row[0])
		#date=str(row[6])
		values.append(row[1]/10000)
		
plt.tight_layout()
sns.set(font_scale=1.20)
plt.figure(figsize=(10,14))
sns.set_style("darkgrid")
#yx = zip(values, labels)
#yx=sorted(zip(values, labels),reverse=True)
#labels = [x for y, x in yx]
#values=[y for y, x in yx]
x_coordinates = np.arange(len(values))
plt.bar(x_coordinates, values, align='center',color='#C0392B',width=1.0)
sns.plt.yticks(weight='bold')
sns.plt.xticks(weight='bold')
plt.xlim(0,501)
#plt.xticks(x_coordinates,labels)
#plt.xticks(rotation=90)
sns.plt.title('No of Question for 500 top tags in SO', weight='bold').set_fontsize('18')
sns.plt.ylabel('Count of Questions in 10^4',weight='bold').set_fontsize('18')
sns.plt.xlabel('Top 500 tags in SO --->',weight='bold').set_fontsize('18')

plt.savefig("top500_bar.png", dpi=500)


