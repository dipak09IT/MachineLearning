# This is example 2 of k nearest
'''
Choosing the Best K
K matters a lot:
K too small (e.g., 1): the model is sensitive to noise and outliers (overfitting).
K too large: the model gets too “blurry” and ignores local patterns (underfitting).
'''
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try different values of K
k_values = range(1, 21)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# Plot accuracy for each K
plt.plot(k_values, scores, marker="o")
plt.xlabel("K (number of neighbors)")
plt.ylabel("Accuracy")
plt.title("Finding the best K")
plt.show()

best_k = k_values[scores.index(max(scores))]
print("Best K:", best_k)

