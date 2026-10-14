# KNN is based on simple idea :"you are similar to the people closest to you"
'''
we want to guess if a fruit is apple or orange based on weight and size,
we insert some data and classify them as apple and orange  then train
'''
import numpy as np
from collections import Counter

# Trainig DATA: [weight(gm), size(cm)]
X_train = np.array([
                   [150, 7], [160, 7.5], [155, 7.2],
                   [130, 6], [120, 5.5], [125, 5.8],
                   ])
y_train = ["Apple", "Apple", "Apple", "Orange", "Orange", "Orange"]


def knn_predict(new_point, X_train, y_train, k=3):
    # step 1: Distance from new point to every training point
    distance = [np.sqrt(np.sum((new_point - x) ** 2)) for x in X_train]

    #step 2: indices of the k smallest distances
    nearest_idx = np.argsort(distance)[:k]

    # step 3: labels of those neighbors
    nearest_labels = [y_train[i] for i  in nearest_idx]

    # Step 4: majority Vote
    return Counter(nearest_labels).most_common(1)[0][0]

new_fruit = np.array([145,6.9])
print("Prediction:", knn_predict(new_fruit, X_train,y_train, k=3))
