import numpy as np
import pandas as pd
import warnings
import os.path
import logging
logging.basicConfig(filename="process.log", level=logging.INFO)

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from app.machine_learning.data_preprocessor import DataPrerocessor

news_labels = {1: "Satire", 2: "Hoax", 3: "Propaganda", 4: "Trusted"}
politifact_labels = {0: "True", 1: "Mostly True", 2: "Half True", 3: "Mostly False", 4: "False", 5: "Pants-on-fire"}

class Classifier:
    classifier = None
    app = None
    model = None
    tfidf_vect_ngram = None

    def __init__(self, app, model, learning_rate=5e-5, max_iters=1000, add_intercept=True):
        try:
            logging.info("Initializing classifier")
            if not app:
                raise Exception("App name must be specified")
            if not model:
                raise Exception("Classifier model must be specified")

            if model.lower() == "maxent":
                if app.lower() == "newsreliability":
                    if os.path.isfile("maxent-news.pkl"):
                        self.classifier = joblib.load("maxent-news.pkl")
                    if os.path.isfile("tfidf-maxent-news.pkl"):
                        self.tfidf_vect_ngram = joblib.load("tfidf-maxent-news.pkl")
                elif app.lower() == "predictingtruthfullness":
                    if os.path.isfile("maxent-politifact.pkl"):
                        self.classifier = joblib.load("maxent-politifact.pkl")
                    if os.path.isfile("tfidf-maxent-politifact.pkl"):
                        self.tfidf_vect_ngram = joblib.load("tfidf-maxent-politifact.pkl")

                if not self.classifier:
                    self.classifier = LogisticRegression(max_iter=max_iters, fit_intercept=add_intercept, multi_class="multinomial", solver="newton-cg")
            elif model.lower() == "multinomialnb":
                if os.path.isfile("mnb-politifact.pkl"):
                    self.classifier = joblib.load("mnb-politifact.pkl")
                if os.path.isfile("tfidf-maxent-politifact.pkl"):
                    self.tfidf_vect_ngram = joblib.load("tfidf-maxent-politifact.pkl")

                if not self.classifier:
                    self.classifier = MultinomialNB()

            self.app = app
            self.model = model

            logging.info("Classifier created")
        except Exception as e:
            logging.exception(str(e))

    def train(self):
        try:
            logging.info("initializing classifier training")
            if self.app.lower() == "newsreliability":
                logging.info("Loading training and test data")
                train_data = pd.read_csv('newsfiles/fulltrain.csv', header=None)
                test_data = pd.read_csv('newsfiles/balancedtest.csv', header=None)

            logging.info("Splitting training and test label and text")
            train_X = train_data[1]
            train_Y = train_data[0]
            test_X = test_data[1]
            test_Y = test_data[0]

            logging.info("Creating TF-IDF N-gram vector")
            self.tfidf_vect_ngram, tfidf_ngram_train, tfidf_ngram_test = DataPrerocessor.generate_tfidf_ngrams(train_X, test_X, 100)

            logging.info("Training classifier")
            self.classifier.fit(tfidf_ngram_train, train_Y)
            
            logging.info("Get classifier accuracy")
            predictions = self.classifier.predict(tfidf_ngram_test)
            result = metrics.accuracy_score(predictions, test_Y)

            logging.info("Saving classifier")
            if self.classifier:
                if self.model.lower() == "maxent":
                    if self.app.lower() == "newsreliability":
                        joblib.dump(self.classifier, "maxent-news.pkl")
                        joblib.dump(self.tfidf_vect_ngram, "tfidf-maxent-news.pkl")
                    elif self.app.lower() == "predictingtruthfullness":
                        joblib.dump(self.classifier, "maxent-politifact.pkl")
                        joblib.dump(self.tfidf_vect_ngram, "tfidf-maxent-politifact.pkl")
                elif self.model.lower() == "multinomialnb":
                    joblib.dump(self.classifier, "mnb-politifact.pkl")
                    joblib.dump(self.tfidf_vect_ngram, "tfidf-mnb-politifact.pkl")

            logging.info("Finish training classifier")
            return result
        except Exception as e:
            logging.exception(str(e))

    def predict(self, X):
        try:
            if self.classifier == None or self.tfidf_vect_ngram == None:
                self.train()

            logging.info("Predicting")
            test_X = self.tfidf_vect_ngram.transform(X)

            result = self.classifier.predict(test_X)

            if self.app.lower() == "newsreliability":
                return news_labels[result[0]]
            elif self.app.lower() == "predictingtruthfullness":
                return politifact_labels[result[0]]
        except Exception as e:
            logging.exception(str(e))



def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Here we use np.exp() as exponential of Euler's

class Model:
    theta = None # theta attribute of the model
    accuracy = 0 # Accuracy of the trained model
    
    # Constructor of our model
    """
    initialize a few attributes for later use
    learning_rate = learning_rate, how should the model change every iteration
    max_iters = max iteration when training
    add_intercept = add intercept to the training and test dataset
    """
    def __init__(self, learning_rate=5e-5, max_iters=1000, add_intercept=True):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.add_intercept = add_intercept

    # Here's the train method
    """
    X = features
    y = class
    max_iter (optional) = maximum iteration we want
    """
    def train(self, X, y, maxIter=100000):
        new_X = X.copy() # Get copy of X
        logging.info("Training model...")

        # Initialize theta attribute with matrix containing 1
        # and make it ((features columns + 1) x 1) size
        self.theta = np.ones((new_X.shape[1] + 1, 1))

        if self.add_intercept:
            intercept = np.ones((len(new_X), 1)) # Create matrix containing intercept with (training rows x 1) size
            new_X = np.hstack((intercept, new_X)) # Concatenate the intercept with the new_X

        # Loop until reach maximum iteration count
        for i in range(maxIter):
            p = sigmoid(new_X.dot(self.theta)) # Create the probability matrix
            gradient = new_X.T.dot(y-p) # Compute the gradient
            self.theta -= self.learning_rate * gradient # Update our model weights

            logging.info("Iteration " + str(i) + " weights: {}".format(self.theta))

        # Indicate training completed and logging.info weight results
        logging.info("Training completed\n")
        logging.info("Weights: {}\n".format(self.theta))

    # Here's the predict method
    def predict(self, X):
        new_X = X.copy() # Get copy of X

        # Return warning if the trained features and the current features do not have the same size
        if not (new_X.shape[1]+1) == len(self.theta):
            return warnings.warn('The model trained with ' + str(len(self.theta)) + ' features')

        if self.add_intercept:
            intercept = np.ones((len(new_X), 1)) # Create matrix containing intercept with (training rows x 1) size
            new_X = np.hstack((intercept, new_X)) # Concatenate the intercept with the new_X
        
        predictions = np.hstack(np.round(sigmoid(new_X.dot(self.theta)))) # Predict the test dataset

        # Return as dataframe
        pred_dataframe = pd.DataFrame({'labels':predictions})
        return pred_dataframe


    def get_accuracy(self, predicted_Y, actual_Y):
        # Compute the accuracy
        predictions = np.hstack(predicted_Y.values)
        actual = np.hstack(actual_Y.values)
        
        return (predictions == actual).sum() / len(predictions)