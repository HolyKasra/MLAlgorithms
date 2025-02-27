import numpy as np
###############################################################################################
###############################################################################################
class DecisionStump:

    def __init__(self):
        self.threshold = None
        self.polarity = 1
        self.feature = None

    def fit(self, X, y, weights):

        n_samples, n_features = X.shape
        min_error = float('inf')

        for featidx in range(n_features):

            thresholds = self._thresh(X[:, featidx])

            for thresh in thresholds:
                
                for polarity in [1, -1]:
                    
                    # initialzing the predictions with 1s
                    predictions = np.ones(n_samples)
                    predictions[polarity * X[:, featidx] < polarity * thresh] = -1

                    error = np.sum(weights[predictions!=y])
                    # the interesting point is that the polarity might have 
                    # significant effect on the error of the classification
                    # since we have samples with different weights!
                    if error < min_error:
                        self.feature = featidx
                        self.threshold  = thresh
                        min_error    = error
                        self.polarity = polarity

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.ones(n_samples)
        y_pred[self.polarity * X[:, self.feature] < self.polarity * self.threshold] = -1
        return y_pred

    def _thresh(self, x):
        x = np.unique(x)
        return np.array([0.5*x[i-1]+0.5*x[i] for i in range(len(x)-1)])

###############################################################################################
###############################################################################################
class AdaBoostClassifier:
    ''' 
    this class is written or binary classification problem. 
    As a result, the labels are 0s or 1s.
    Since this classifier does not handle encoding, the necessary preprocessing steps
    should take place before initiating this class!
    '''
    def __init__(self, n_estimators=50, learning_rate=1, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        

    def fit(self, X, y):

        # Initializing the parameters!
        n_samples = X.shape[0]

        # Sequential classifiers container
        self.classifiers = [] 

        # information_index of each classifier
        self.alpha_vals = []

        # Samples weight initialization
        weights = np.ones(n_samples,dtype=np.float64)/n_samples

        for _ in range(self.n_estimators):
            
            # initializing a classification tree with max_depth=1 (Actually an stump!)
            clf = DecisionStump()
            clf.fit(X, y, weights=weights)

            # Modifying the prediction output!
            y_pred = clf.predict(X)

            # Updating the parameters
            alpha, weights = self._update_stumps(y_true=y, y_pred=y_pred, weights=weights)

            # Logging the parameters!
            self.alpha_vals.append(alpha)
            self.classifiers.append(clf)
    
    def predict(self, X):
        final_preds = np.zeros(X.shape[0])

        for alpha_val, clf in zip(self.alpha_vals,self.classifiers):
            final_preds += alpha_val * clf.predict(X)
        
        return np.sign(final_preds)
        
    def _update_stumps(self, y_true, y_pred, weights, epsilon = 1e-10):

        # Misclassification index!
        incorrect = y_true != y_pred

        # Obtaining error_rate
        error_rate = np.sum(weights[incorrect])

        # Making sure error_rates stays in [0,1] interval.
        error_rate = np.clip(error_rate, epsilon, 1-epsilon)

        # alpha parameter for each classifier!
        alpha = 0.5 * self.learning_rate * np.log((1-error_rate)/error_rate)

        # Calculating the weights for each data observation
        weights *= np.exp(-alpha * y_true * y_pred)
        weights /= np.sum(weights)
        return alpha, weights
    
###############################################################################################
###############################################################################################
if __name__ == '__main__':

    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_classes=2, 
                               n_samples=1000, 
                               n_features=5, 
                               n_informative=2, 
                               random_state=42)
    y[y==0]=-1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    ada = AdaBoostClassifier(n_estimators=10)
    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'My AdaBoost Classifier Accuracy: {acc}')
