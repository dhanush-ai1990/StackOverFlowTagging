from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np

[X,y]=joblib.load('Text25Vectors.pkl')

print (len(X))
print X.shape
print (len(y))
X=np.array(X)
y=np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=100,n_jobs=6)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_text25_pipeline.py')