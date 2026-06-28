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
    
    # Converting image RGB matrices into greyscale 1-D arrays using image processing function 

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
    xTestNormalized = (fireTesting - training_mean) / training_std # Normalizing the training data 
    yTest = np.array([1]*len(fireTest) + [0]*len(noFireTest)) # Test Label's

    # Validation data 
    xValid = np.array(fireValidation + noFireValidation)
    xValidNormalized = (xValid - training_mean) / training_std # Normalizing the training data 
    yValid = np.array( [1]*len(fireValidation) + [0]*len(noFireValidation) ) # Test Label's

    
    # Gaussian Naive Bayes Model Training:

    gnb = GaussianNB() # Creating a Gaussian Naive Baye's Object 

    # Training the model 
    gnb.fit(xTrainNormalized, yTrain)

    # Predict
    yPredGnbValid = gnb.predict(xValidNormalized)

    # Prediction for the test  
    yPredGnbTest = gnb.predict(xTestNormalized)


    # Classification report 
    reportGnbValid = classification_report(yValid, yPredGnbValid,labels=[0,1],target_names=["Wildfire", "No Wildfire"])

    reportGnbTest = classification_report(yTest, yPredGnbTest,labels= [0,1],target_names=["Wildfire", "No Wildfire"]) # Only useful for determing if we're overfitting 

    print("Validation Summary:")
    print(reportGnbValid)
    print("Test Summary:")
    print(reportGnbTest)

if __name__ == "__main__":
    main()
