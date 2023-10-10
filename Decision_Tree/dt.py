from typing import List

class Node:
    def __init__(self, feature_index: int = None, threshold: float = None, value: float = None,
                 left: 'Node' = None, right: 'Node' = None, info_gain: float = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.info_gain = info_gain
    
class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: List[List[float]], y: List[int]):
        dataset = []
        for i in range(len(X)):
            dataset.append(X[i] + [y[i]])
        self.root = self.build_tree(dataset)
        return self.root

    def predict(self, X: List[List[float]]):
        predictions = []
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x: List[float], tree: Node):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


    def build_tree(self, dataset: List[List[float]], curr_depth: int = 0):
        X = [row[:-1] for row in dataset]
        Y = [row[-1] for row in dataset]
        num_samples, num_features = self.shape(X)
        if num_samples >= 0 and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left=left_subtree, right=right_subtree, info_gain=best_split["info_gain"])
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def shape(self, dataset: List[List[float]]):
        sample = len(dataset)
        feature = len(dataset[0])
        return sample, feature
    
    def unique(self, feature: List[float]):
        unique = []
        for i in feature:
            if i not in unique:
                unique.append(i)
        return unique
    
    def log2(self, x):
        if x <= 0:
            return float('nan')
        elif x == 1:
            return 0
        else:
            count = 0
            while x > 1:
                x /= 2
                count += 1
            return count
    
    def split(self, dataset: List[List[float]], feature_index: int, threshold: float):
        dataset_left = []
        dataset_right = []
        for row in dataset:
            if row[feature_index] <= threshold:
                dataset_left.append(row)
            else:
                dataset_right.append(row)
        return dataset_left, dataset_right
    
    def entropy(self, y: List[int]):
        labels = self.unique(y)
        entropy = 0
        for label in labels:
            p_cls = len([value for value in y if value == label]) / len(y)
            entropy += -p_cls * self.log2(p_cls)
        return entropy
    
    def gini_index(self, y: List[int]):
        labels = self.unique(y)
        gini = 0
        for label in labels:
            p_cls = len([value for value in y if value == label]) / len(y)
            gini += p_cls ** 2
        return 1 - gini
    
    def information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = self.gini_index(parent) - (weight_left * self.gini_index(left_child) + weight_right * self.gini_index(right_child))
        return gain
    
    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {
            "feature_index": None,
            "threshold": None,
            "dataset_left": None,
            "dataset_right": None,
            "info_gain": -float("inf")
        }
        max_info_gain = -float("inf")
        for feature_index in range(num_features):
            feature_values = [row[feature_index] for row in dataset]
            unique_values = self.unique(feature_values)
            for threshold in unique_values:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y = [row[-1] for row in dataset]
                    left_y = [row[-1] for row in dataset_left]
                    right_y = [row[-1] for row in dataset_right]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent: str = " "):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    