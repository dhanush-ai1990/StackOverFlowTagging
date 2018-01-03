import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.externals import joblib
import pandas as pd
import math
import datetime
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
'''
dict1=joblib.load("tag_20_count.pkl")
labels=list(dict1.keys())
values=list(dict1.values())
sns.set()
sns.set_style("dark",{"xtick.major.size": 4,"ytick.major.size": 4})
'''
filename='GrowthofUsers.csv'
chunksize = 10 ** 6
values=[]
months=[]
year=[]
dates=[]
for chunk in pd.read_csv(filename, chunksize=chunksize):
	rows = chunk.values.tolist()
	for row in rows:
		yr = str(row[0])[2:]
		mon=datetime.date(1900, row[1], 1).strftime('%b')
		date=str(mon)+ "'"+yr
		dates.append(date)
		year.append(row[0])
		months.append(row[1])
		#date=str(row[6])
		values.append(row[2])
		
#yx = zip(values, labels)
#yx=sorted(zip(values, labels),reverse=True)
#labels = [x for y, x in yx]
#values=[y for y, x in yx]
values=values[0:-2]
sns.set(font_scale=1.35)
plt.figure(figsize=(10,10))
sns.set_style("darkgrid")
ax =plt.axes()
x_coordinates = np.arange(len(values))

plt.bar(x_coordinates, values, color='#C0392B',width=1.0)
sns.plt.xticks(x_coordinates,dates,weight='bold')
sns.plt.yticks(weight='bold')
plt.xticks(rotation=90)
sns.plt.title('Growth of new users in SO', weight='bold').set_fontsize('18')
sns.plt.ylabel('Number of new users',weight='bold').set_fontsize('18')
sns.plt.xlabel('Months from 2008 to present',weight='bold').set_fontsize('18')
plt.xlim(0,113)
n=2
[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i % 4 != 0)]
#[l.set_visible(True) for (i,l) in enumerate(ax.xaxis.get_ticklabels())if i % n != 0]
#ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#plt.xaxis.set_major_formatter(majorFormatter)
plt.savefig("UsersGrowth_bar.png",dpi=500)




