import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.externals import joblib



dict1=joblib.load("tag_20_count.pkl")
labels=list(dict1.keys())
values=list(dict1.values())
sns.set()
sns.set_style("dark",{"xtick.major.size": 4,"ytick.major.size": 4})

plt.figure(figsize=(16, 12))
yx = zip(values, labels)
yx=sorted(zip(values, labels),reverse=True)
labels = [x for y, x in yx]
values=[y for y, x in yx]
x_coordinates = np.arange(len(values))
plt.bar(x_coordinates, values, align='center',color='#A93226')
#plt.xticks(x_coordinates,labels)
#plt.xticks(rotation=90)
plt.title("# of Question for 20 selected languages")
plt.ylabel('count ')
plt.savefig("top20_bar.png", dpi=300)