# Train a logistic regression classifierr to predict  whether a flower is IRIS VIRGINICA or NOT

import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


"""
#print(list(iris.keys()))

# to show iris data 
print(iris['data'])

# to check shape of iris 
print(iris['data'].shape)

# To find target of given data, We do-
print(iris['target'])

# To give description of given target data we do-
print(iris['DESCR']) 

"""

# using one feature "petalwidth " form iris dataset and training logistic regression

# we can store data and target in single variable
X = iris["data"][:, 3:]    #slicing is done (need third column)
Y = (iris["target"] == 2).astype(np.int64)    # we need number not true/false

# Train a logistic refgression classifier
clf =LogisticRegression()
clf.fit(X,Y)

example = clf.predict(([[2.6]]))
print(example)



# using matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1,1)
Y_prob = clf.predict_proba(X_new)            # predict probability
plt.plot(X_new, Y_prob[:,1], "g-", label="virginica")
plt.show()


# print(Y)
# print(iris["data"])
# print(X)
