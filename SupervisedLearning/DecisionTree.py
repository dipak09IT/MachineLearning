'''import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# data set using - iris classification

data = pd.read_csv("Iris.csv")

X = data.drop(columns["id", "Species"]).values
y = data["Species"].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 123)

class Node:
    def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None, right=None, value=None):

        # Decision Node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right

        # Leaf Node
        self.value = value

# tree building and logic implementation

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # recursive function
        def build_tree(self, dataset, curr_depth=0):
            x, y = dataset[:, :-1],dataset[:,-1]
            n_samples, n_features = X.shape

            if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
                best_split = self.best_split(dataset, n_features)

                if best_split["info_gain"] > 0:
                    left_node = self.build_tree(best_split["left_dataset"], curr_depth + 1)
                    right_node = self.build_tree(best_split["right_dataset"], curr_depth + 1)

                    return Node(best_split["feature_idx"], best_split["threshold"], best_split["info_gain"], left_node, right_node )

            leaf_value =  Counter(y).most_common(1)[0][0]
            return Node(value = leaf_value)

        def best_split(self, dataset, n_features):
            best_split= {'feature_idx':None,'threshold': None, 'info_gain': -1, 'left_dataset': None,'right_dataset': None}

            for feature_idx in range(n_features):
                feature_values = dataset[:, feature_idx]
                thresholds = np.unique(feature_values) 

                for threshold in thresholds:
                    left_dataset, right_dataset = self.split(dataset, feature_idx, threshold) 

                    if len(left_dataset) and len(right_dataset):
                        parent_y, left_y, right_y = dataset[:, -1], right_dataset[:, -1]

                        info_gain = self.information_gain(parent_y, left_y, right_y)

                        if info_gain > best_split['info_gain']:
                            best_split['feature_idx'] = feature_idx
                            best_split['threshold'] = threshold
                            best_split['info_gain'] = info_gain
                            best_split['left_dataset'] = left_dataset
                            best_split['right_dataset'] = right_dataset

        return best_split 

    def split(self, dataset, feature_idx, threshold):
        left_dataset = np.array([row for row in dataset if row[feature_idx] <= threshold])
        right_dataset = np.array([ row for row in dataset if row[feature_idx] > threshold])

        return left_dataset, right_dataset
    
    def information_gain(self, parent_y, left_y, right_y):
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)

        information_gain = self.entropy(parent_y) - (left_weight * self.entropy(left_y) + right_weight * self.entropy(right_y))
        return information_gain
    
    def entropy(self, y):
        self.entropy = 0

        class_labels = np.unique(y)
        for class_label in class_labels:
            p = len(y[y == class_label]) / len(y)
            entropy += -p * np.log2(p)

        return entropy

    def fit(self, X,y):
        dataset = np.concatenate([X,y.reshape(-1,1)], axis =1)
        self.root =  self.build_tree(dataset)

    def predict(self, X):
        predictions = [self.predict_class(row, self.root) for row in X]
        return predictions
    
    def predict_class(self, row, node):
        if node.value != None:
            return node.value
        
        feature_val = row[node.feature_idx]
        if feature_val <= node.threshold:
            return self.predict_class(row, node.left)
        else:
            return self.predict_class(row, node.right)

    def print_tree(self , node=Node, depth = 0, indent = "| " ):
        prefix = indent * depth

        if node is None:
            node = self.root

        if node.value is not None:
            print(f"{prefix} |---class: {node.value}")
            return

        feature_label = f"Feature {node.feature_idx}"

        print(f"{prefix}---{feature_label} <= {node.threshold}")
        self.print_tree(node.left, depth +1, indent)

        print(f"{prefix}|---{feature_label} > {node.threshold}")
        self.print_tree(node.right, depth + 1, indent)

        dt = DecisionTree(min_samples_split = 2, max_depth = 2)
        dt.fit(x_train, y_train)
        prediction = dt.predict(x_test)

        accuracy = np.mean(prediction == y_test) * 100
        print(f"Accuracy: {accuracy:2f}%")

        dt.print_tree()

'''


# This code builds a Decision Tree classifier from scratch (without sklearn's DecisionTreeClassifier) and tests it on the Iris dataset.


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# Dataset: Iris classification
data = pd.read_csv("Iris.csv")

X = data.drop(columns=["Id", "Species"]).values
y = pd.factorize(data["Species"])[0]      # convert species names to 0, 1, 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


class Node:
    def __init__(self, feature_idx=None, threshold=None, info_gain=None,
                 left=None, right=None, value=None):
        # Decision node
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right
        # Leaf node
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    # Recursive tree building
    def build_tree(self, dataset, curr_depth=0):
        x, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = x.shape

        if n_samples >= self.min_samples_split and curr_depth < self.max_depth:
            best_split = self.best_split(dataset, n_features)

            if best_split["info_gain"] > 0:
                left_node = self.build_tree(best_split["left_dataset"], curr_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], curr_depth + 1)
                return Node(best_split["feature_idx"], best_split["threshold"],
                            best_split["info_gain"], left_node, right_node)

        leaf_value = int(Counter(y).most_common(1)[0][0])
        return Node(value=leaf_value)

    def best_split(self, dataset, n_features):
        best_split = {"feature_idx": None, "threshold": None, "info_gain": -1,
                      "left_dataset": None, "right_dataset": None}

        for feature_idx in range(n_features):
            feature_values = dataset[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_dataset, right_dataset = self.split(dataset, feature_idx, threshold)

                if len(left_dataset) and len(right_dataset):
                    parent_y = dataset[:, -1]
                    left_y = left_dataset[:, -1]
                    right_y = right_dataset[:, -1]

                    info_gain = self.information_gain(parent_y, left_y, right_y)

                    if info_gain > best_split["info_gain"]:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["info_gain"] = info_gain
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset

        return best_split

    def split(self, dataset, feature_idx, threshold):
        left_dataset = np.array([row for row in dataset if row[feature_idx] <= threshold])
        right_dataset = np.array([row for row in dataset if row[feature_idx] > threshold])
        return left_dataset, right_dataset

    def information_gain(self, parent_y, left_y, right_y):
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)
        return self.entropy(parent_y) - (
            left_weight * self.entropy(left_y) + right_weight * self.entropy(right_y)
        )

    def entropy(self, y):
        entropy = 0
        for class_label in np.unique(y):
            p = len(y[y == class_label]) / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def fit(self, X, y):
        dataset = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        return np.array([self.predict_class(row, self.root) for row in X])

    def predict_class(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self.predict_class(row, node.left)
        return self.predict_class(row, node.right)

    def print_tree(self, node=None, depth=0, indent="|   "):
        if node is None:
            node = self.root
        prefix = indent * depth

        if node.value is not None:
            print(f"{prefix}|--- class: {node.value}")
            return

        label = f"Feature {node.feature_idx}"
        print(f"{prefix}|--- {label} <= {node.threshold}")
        self.print_tree(node.left, depth + 1, indent)
        print(f"{prefix}|--- {label} > {node.threshold}")
        self.print_tree(node.right, depth + 1, indent)


# Run (this must be at the left margin, outside the class)
dt = DecisionTree(min_samples_split=2, max_depth=2)
dt.fit(X_train, y_train)
prediction = dt.predict(X_test)

accuracy = np.mean(prediction == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

dt.print_tree()                                                                                                                 

