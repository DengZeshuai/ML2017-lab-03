import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y,max_depth=1):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.weights_of_samples = np.ones((X.shape[0],1),dtype=float) / X.shape[0]
        self.prediction_of_classifiers = []
        self.alphas_of_classifiers = []
        self.classifiers = []
        
        print("Train start!")
        for m in range(self.n_weakers_limit):
            weights_of_samples = self.weights_of_samples.reshape(X.shape[0])
            
            new_weak_classifier = self.weak_classifier(max_depth=max_depth, splitter="random", random_state=1)
            new_weak_classifier.fit(X,y, sample_weight=weights_of_samples)
            self.classifiers.append(new_weak_classifier)
            
            prediction = new_weak_classifier.predict(X)
            prediction = np.array(prediction).reshape((X.shape[0],1))
            
            #compute error
            error = np.zeros(prediction.shape)
            error[prediction != y] = 1
            error_m = (self.weights_of_samples*error).sum()
            if error_m > 0.5:
                print("Train error: one of error rate less than 0.5")
                break
            
            #compute weight of classifier and samples
            alpha_m = 0.5*np.log((1.0-error_m)/ error_m)
            # print(alpha_m)
            self.alphas_of_classifiers.append(alpha_m)
            Z_m = (self.weights_of_samples*np.exp(-alpha_m*y*prediction)).sum()
            self.weights_of_samples =  self.weights_of_samples/Z_m * np.exp(-alpha_m*y*prediction)
        

    def predict_scores(self, X,y):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        prediction_of_Adaboost = self.predict(X)
        prediction_score_of_Adaboost = prediction_of_Adaboost*y
        prediction_score_of_Adaboost[prediction_score_of_Adaboost==-1] == 0
        
        accuracy = prediction_score_of_Adaboost.sum()/prediction_score_of_Adaboost.shape[0]
        return accuracy

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        prediction_of_Adaboost = np.zeros((X.shape[0],1))
        for i in range(self.n_weakers_limit):
                classifier_predict = np.array(self.classifiers[i].predict(X)).reshape((X.shape[0],1))
                prediction_of_Adaboost += self.alphas_of_classifiers[i]*classifier_predict
        
        # classification
        prediction_of_Adaboost[prediction_of_Adaboost>threshold] = 1
        prediction_of_Adaboost[prediction_of_Adaboost<=threshold] = -1
        return prediction_of_Adaboost

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
