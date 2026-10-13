# working on iris dataset and making decision based on random forest

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier   # ensemble means using deecison trees classifier multiple times


iris = load_iris(as_frame = True)

X= iris.data
y = iris.target

#print("iris data", X) # to print iris data     speal length        petal width  
#print("iris target", y)

# How a random forest is trained
'''
--> rows sampling 
--> growss a tree
--> repeate same process
--> aggegration of predction

'''

print(iris.target_names)     # ['setosa' 'versicolor' 'virginica']

y = y.map(dict(enumerate(iris.target_names)))

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))
print("OOB score:", model.oob_score_)
