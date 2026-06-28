from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import numpy as np
from ImageProcessing import imageProcessing

# Returns a value a optimal C value within the cGrid
def optimalC(trainingData, trainingLabels):

    # Training Logistic Model 
    logreg = LogisticRegression(solver="lbfgs", max_iter=1000)
    cGrid = {"C":[0.001, 0.01, 0.1, 1, 10, 100]}
    
    gridSearch = GridSearchCV(logreg, cGrid, cv = 5, scoring="accuracy")
    
    # Fit the data 
    gridSearch.fit(trainingData, trainingLabels)
    optimalC = gridSearch.best_params_['C']
    return optimalC


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
    
    # Flattening the images into valid data lists
    
    # Training Data: 
    fireTrain = imageProcessing(pathTrainFire)
    noFireTrain = imageProcessing(pathTrainNoFire)

    # Testing Data:
    fireTest = imageProcessing(pathTestFire)
    noFireTest = imageProcessing(pathTestNoFire)

    # Validation Data:
    fireValidation = imageProcessing(pathValFire)
    noFireValidation = imageProcessing(pathValNoFire)

    # Creating data arrays for model training 
    xTrainData = np.array(fireTrain + noFireTrain)

    # Normalizing the data using per-pixel standardization 
    scaler = StandardScaler() # scaler object 
    xTrainNormalized = scaler.fit_transform(xTrainData) # Scaling each pixel position independently 

    # Vector with label's for data 
    yTrain = np.array([1]*len(fireTrain) + [0]*len(noFireTrain))

    # Testing Data: Extracted from the data folders    
    fireTesting = np.array(fireTest + noFireTest)
    xTestNormalized = scaler.transform(fireTesting)
    yTest = np.array([1]*len(fireTest) + [0]*len(noFireTest)) # Test Label's

    # Validation data 
    xValid = np.array(fireValidation + noFireValidation)
    xValidNormalized = scaler.transform(xValid)
    yValid = np.array( [1]*len(fireValidation) + [0]*len(noFireValidation) ) # Test Label's

      # Reducing the amount of features using Principal Component Analysis
    pca = PCA(n_components=100)
    xTrainPCA = pca.fit_transform(xTrainNormalized)
    xValidPCA = pca.transform(xValidNormalized)
    xTestPCA = pca.transform(xTestNormalized)
    
    print(xTrainPCA.shape)
    print(yTrain.shape)
    
    # Final Evaluation 
    optimalC = 0.001
    logreg = LogisticRegression(max_iter=200, C = optimalC)
    logreg.fit(xTrainPCA, yTrain)
    
    # Obtaining prediction on the validation and testing datasets
    yLogPredValid = logreg.predict(xValidPCA)
    yLogPredTest = logreg.predict(xTestPCA)

    # Classification report: Evaluating the model's preformance on the two datasets 
    reportLogRValid = classification_report(yValid,yLogPredValid,labels=[0,1],target_names=["No Wildfire", "Wildfire"])
    reportLogRTrain = classification_report(yTest,yLogPredTest,labels= [0,1],target_names=["No Wildfire", "Wildfire"])

    # Printing a summary of the validation and testing datasets 
    print("Validation Summary:")
    print(reportLogRValid)
    print("Test Summary:")
    print(reportLogRTrain)

if __name__ == "__main__":
    main()
