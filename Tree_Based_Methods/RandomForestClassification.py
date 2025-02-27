from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from classification_trees import ClassificationTree

@dataclass
class RFClassifier:
    n_trees: Optional[int] = 200
    max_depth: Optional[int] = 100
    min_samples_split: Optional[int] = 2
    n_feats: Optional[int] = 2
    classifiers: Optional[list] = field(default_factory=list)

    def fit(self, X, y):

        # initializing a list in order to calculate 
        # the feature importance in random forest classifier
        self.random_importance = np.array([0] * X.shape[1], dtype=np.float64)

        for _ in range(self.n_trees):
            clf = ClassificationTree(max_depth=self.max_depth,
                                     min_samples_split=self.min_samples_split,
                                     n_feats=self.n_feats)
            
            Xboot, yboot = self._bootstrap(X, y)
            clf.fit(Xboot, yboot)
            self.classifiers.append(clf)

    def predict(self,X):

        preds = np.array([clf.predict(X) for clf in self.classifiers])

        # n_rows = number of trees ---> n_cols = number of samples!
        # by doing this, we are putting all tree's votes in a row instead of cols 
        # just to make easier calculations!
        preds = preds.T
        y_pred = np.array([self._aggregate(pred) for pred in preds])
        
        return y_pred

    def _bootstrap(self, X, y):
        idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return X[idx], y[idx]
    
    def _aggregate(self, y):
        counts = Counter(y)
        return counts.most_common(1)[0][0]
###############################################################################################
###############################################################################################
if __name__ == '__main__':
    from collections import Counter
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # My Random Forest
    clf = RFClassifier(n_trees=100, max_depth=10, min_samples_split=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    my_acc = accuracy_score(y_test, y_pred) * 100
    print(f'Accuracy of My RFClassifier is about: {my_acc:.3f}%')

    # Scikit-Learn Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    sk_accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'Accuracy of Scikit-Learn RandomForest Classifier is about: {sk_accuracy:.3f}%')