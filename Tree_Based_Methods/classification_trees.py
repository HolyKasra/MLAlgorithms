from dataclasses import dataclass
import numpy as np
from collections import Counter
from typing import Optional

@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[int] = None
    gini: Optional[float] = None
    n_samples: Optional[int] = None
    # by selecting this attribute, the number for each class 
    # at each node is shown. ---> node.right.members = [42,43,12]
    members: Optional[np.ndarray] = None 

@dataclass
class ClassificationTree:
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    n_feats: Optional[int] = None
    root: Optional[Node] = None

    def fit(self, X, y):
        
        # Selecting the specified number of features!
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])

        # Saving class labels to keep track of class numbers later!
        self.labels = np.unique(y)
        
        # growing tree from the root recursively!
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):

        # Number of available classes!
        n_class   = len(np.unique(y)) 
        n_samples = len(y)
        
        # Calculate current Gini index for this node
        best_feature, best_threshold, best_gain, parent_gini = self._best_split(X, y)
        # Keeping track of class members!
        current_members = [np.sum(y==label) for label in self.labels]

        # Stoping criteria for training of the decision tree
        if (
            self.max_depth is not None and depth >= self.max_depth or \
            n_samples < self.min_samples_split or \
            n_class == 1
        ):
            leaf_value = self._majority_vote(y)
            return Node(value=leaf_value, 
                        gini=parent_gini, 
                        n_samples=n_samples, 
                        members=current_members)

        # the situation in which the gini of the parent node is less than its children's
        if best_feature is None:
            leaf_value = self._majority_vote(y)
            return Node(value=leaf_value, 
                        gini=parent_gini, 
                        n_samples=n_samples, 
                        members=current_members)
        
        best_left = X[:, best_feature] < best_threshold
        best_right = ~best_left
        
        left = self._grow_tree(X[best_left], y[best_left], depth+1)
        right = self._grow_tree(X[best_right], y[best_right], depth+1)
        
        return Node(feature=best_feature, 
                    threshold=best_threshold, 
                    left=left, 
                    right=right, 
                    gini=parent_gini, 
                    n_samples=n_samples, 
                    members=current_members)

    def _best_split(self, X, y):
        best_gain = -1
        best_threshold = None
        best_feature = None

        # choosing random features to train the tree!
        featidxs = np.random.choice(X.shape[1], self.n_feats)
        
        # Gini parent or total gini of a node!
        parent_gini = self._gini(y)

        for featidx in featidxs:
            thresholds = self._thresh(np.unique(X[:,featidx]))

            for threshold in thresholds:
                left_mask = X[:,featidx] < threshold
                right_mask = ~left_mask

                left_size, right_size = np.sum(left_mask), np.sum(right_mask)

                # it means that the node cannot be devided!
                if (left_size == 0 or right_size == 0):
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # Gini coefficient
                gini_left, gini_right = self._gini(y_left), self._gini(y_right)
                weighted_gini = (left_size * gini_left + right_size * gini_right) / len(y)
                gini_gain = parent_gini - weighted_gini

                # Keeping  track of the best measures!
                if gini_gain > best_gain:
                    best_gain = gini_gain 
                    best_threshold = threshold 
                    best_feature = featidx
                         
        return best_feature, best_threshold, best_gain, parent_gini
    
    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)
        
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)
    
    def _majority_vote(self, y):
        counts = Counter(y)
        return counts.most_common(1)[0][0]
    
    def _thresh(self, x):
        return np.array([0.5 * x[i] + 0.5 * x[i+1] for i in range(len(x)-1)])
###############################################################################################
###############################################################################################
if __name__ == '__main__':
    
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn import tree

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # My implementation!
    clf_tree = ClassificationTree()
    clf_tree.fit(X_train,y_train)

    y_pred_custom = clf_tree.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)

    print(f'Members of root->right->right are: {clf_tree.root.right.right.members}')
    print(f'Members of root->right->right->left are: {clf_tree.root.right.right.left.members}')

    # Scikit-Learn Implementation
    sk_tree = tree.DecisionTreeClassifier()
    sk_tree.fit(X_train, y_train)

    y_pred = sk_tree.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred)

    print(f'My Model Accuracy:{acc_custom*100:>11}\nScikit-Learn Accuracy:{acc_sklearn*100:^10}')


