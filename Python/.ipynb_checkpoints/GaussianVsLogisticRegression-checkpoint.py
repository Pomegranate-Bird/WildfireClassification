from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import numpy as np
from ImageProcessing import imageProcessing



def main():

    # Paths: Training data 
    pathTrainFire = ("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/train/fire")
    pathTrainNoFire = ("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/train/nofire")

     # Testing Data: 
    pathTestFire = ("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/test/fire")
    pathTestNoFire = ("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/test/nofire")

    # Validation Data: 
    pathValFire = ("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/val/fire")
    pathValNoFire = ("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/val/nofire")
    

    # Training Data: Obtained from flattening all the training images 
    fireTrain = imageProcessing(pathTrainFire)
    noFireTrain = imageProcessing(pathTrainNoFire)

    # Testing Data: Obtained from flattening all the training images 
    fireTest = imageProcessing(pathTestFire)
    noFireTest = imageProcessing(pathTestNoFire)

    # Validation Data: Obtained from flattening all the training images 
    fireValidation = imageProcessing(pathValFire)
    noFireValidation = imageProcessing(pathValNoFire)


    # Training data: Extracted from the data folders
    xTrainData = np.array(fireTrain + noFireTrain)

    # Mean and standard deviation of the training Data 
    training_mean = np.mean(xTrainData)
    training_std = np.std(xTrainData)

    # Normalizing the training data 
    xTrainNormalized = (xTrainData - training_mean) / training_std

    # Vector with label's for data 
    yTrain = np.array([1]*len(fireTrain) + [0]*len(noFireTrain))

    # Testing Data: Extracted from the data folders    
    fireTesting = np.array(fireTest + noFireTest)
    xTest = (fireTesting - training_mean) / training_std # Normalizing the training data 
    yTest = np.array([1]*len(fireTest) + [0]*len(noFireTest)) # Test Label's

    # Validation data 
    xValid = np.array(fireValidation + noFireValidation)
    xValidNormalized = (xValid - training_mean) / training_std # Normalizing the training data 
    yValid = np.array( [1]*len(fireValidation) + [0]*len(noFireValidation) ) # Test Label's

    
    # Checking the Dimension's of the Data
    # print("xTrain:", xTrainNormalized.shape)
    # print("yTrain:", yTrain.shape)

    # print("xTest:", xValid.shape)
    # print("yTest:", yValid.shape)

    # Logistic Regression Model Training 

    logreg = LogisticRegression(max_iter = 1000) # Logistic Regression Object 

    logreg.fit(xTrainNormalized, yTrain) # Training the Logistic Regression 

    yLogPredValid = logreg.predict(xValidNormalized)
    yLogPredTrain = logreg.predict(xTrainNormalized)


    # Obtaining Logistic Regression Validation Score 
    validationScoreLogR = metrics.accuracy_score(yValid, yLogPredValid)

    # Gaussian Naive Bayes Model Training:

    gnb = GaussianNB() # Creating a Gaussian Naive Baye's Object 

    # Training the model 
    gnb.fit(xTrainNormalized, yTrain)

    # Predict the labels for the test set 
    yPredGnbValid = gnb.predict(xValidNormalized)
    yPredGnbTrain = gnb.predict(xTrainNormalized)


    # Classification report 

    reportLogRValid = classification_report(yValid,yLogPredValid,labels=[0,1],target_names=["Wildfire", "No Wildfire"])
    reportGnbValid = classification_report(yValid, yPredGnbValid,labels=[0,1],target_names=["Wildfire", "No Wildfire"])
    reportLogRTrain = classification_report(yTrain,yLogPredTrain,labels= [0,1],target_names=["Wildfire", "No Wildfire"])
    reportGnbTrain = classification_report(yTrain, yPredGnbTrain,labels= [0,1],target_names=["Wildfire", "No Wildfire"])

    print("Validation Summary:")
    print(reportLogRValid)
    print(reportGnbValid)
    print("Training Summary:")
    print(reportLogRTrain)
    print(reportGnbTrain)

if __name__ == "__main__":
    main()
