## numpy is used for creating fake data
import numpy as np 
import matplotlib as mpl 
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.externals import joblib
import seaborn as sns
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1



## agg backend is used to create plot as a .png file
mpl.use('agg')

import matplotlib.pyplot as plt 
sns.set(font_scale=1.1)

#arr = []
data_to_plot = []

np.random.seed(10)
total_400_lines = 0
code_loc='/Users/Dhanush/Desktop/SODATAFINAL/code25/'
name_file=['javascript','sql','java','c#','python','php','c++','c','typescript','ruby','swift','objective-c','vb.net','assembly','r','perl','vba','matlab','go','scala','groovy','coffeescript','lua','haskell']
"""
for item in name_file:
    code_loc_current=code_loc+item+'/'
    print(code_loc_current)
    file_list = glob.glob(os.path.join(code_loc_current, "*.txt"))
    Maxnumber = 0;
    arr = []
    for file_path in file_list:
        with open(file_path) as f_input:
            num_of_line = 0
            for num_of_line, l in enumerate(f_input):
                pass
            if num_of_line > 200:
               total_400_lines = total_400_lines + 1
               #print(num_of_line)
               #print(file_path)
               #print('----------------------------')
               continue
            if num_of_line == 0:
                continue
            #Maxnumber = Maxnumber + 1
            #if Maxnumber == 50000:
                #print(Maxnumber)
            #    break
            #print(num_of_line)
            arr.append(num_of_line)
    #print(arr)
    collectn = np.random.normal(arr)
    data_to_plot.append(collectn)
    #print(arr)
print(total_400_lines)


from sklearn.externals import joblib
joblib.dump(data_to_plot,"data_to_plot.pkl")
"""
## Create data
#np.random.seed(10)
#collectn_1 = np.random.normal(arr)
#collectn_2 = np.random.normal(arr2)
#print(arr)
#print(arr2)
#collectn_3 = np.random.normal(90, 20, 200)
#collectn_4 = np.random.normal(70, 25, 200)

## combine these different collections into a list    
#data_to_plot = [collectn_1, collectn_2]

## Custom x-axis labels
data_to_plot=joblib.load("data_to_plot.pkl")
# Create a figure instance
fig = plt.figure(1, figsize=(13, 6))



sns.set_style("dark")
sns.set_style("white",{"xtick.major.size": 10,"ytick.major.size": 10})
sns.plt.title('The Length of Code Snippets Extracted From Stack OverFlow Question Posts',weight='bold').set_fontsize('14')
sns.plt.ylabel('Number of Lines of Code',weight='bold').set_fontsize('12')
sns.plt.xlabel('Programming Languages',weight='bold').set_fontsize('12')

# Create an axes instance
#ax = fig.add_subplot(111)
#ax.set_axisbelow(True)

# Create the boxplot
#plt.boxplot(data_to_plot)
sns.boxplot(data=data_to_plot)
plt.tight_layout()
name_file=[t.title() for t in name_file]
plt.xticks([0,1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], name_file,ha='right',weight='bold')
plt.xticks(rotation=45)
plt.yticks([x for x in range(0,210,10)])
plt.ylim(-10,210)
plt.gcf().subplots_adjust(bottom=0.20)
# Save the figure
#plt.show()
plt.savefig('boxplot.png',dpi=300)


