# import libraries
import tensorflow as tf
from tensorflow.keras.layers import Dropout # Prevent Overfitting since I have a small dataset
import numpy as np
from tensorflow.keras.models import Sequential # Model 
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation

# Metrics 
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Logistic Regression 
from sklearn.linear_model import LogisticRegression

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Utils 
from pathlib import Path
import cv2
import numpy as np

# Helper Function for converting images to RGB: Deep Learning CNN 
def obtainData(path:str):
    reshapedImages = []
    for image_path in path.rglob("*.jpg"):
        img = cv2.imread(str(image_path)) # Obtaining Image 
        # Skip Invalid Images 
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Obtaining RGB Image
        img = cv2.resize(img, (64,64)) # Uniform Dimensions for all data 
        reshapedImages.append(img)
    return reshapedImages


# Extracting Training Data
root1 = Path("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/train/fire")
root2 = Path("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/train/nofire")

# Gathering RGB training images from paths
fireTrainingData = obtainData(root1) # Extracting the training images containing wildfires
noFireTrainingData = obtainData(root2) # Extracting the training images containing no wildfires
trainingData = np.array(fireTrainingData+noFireTrainingData) # Full training dataset

# Extracting validation Data
root3 = Path("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/val/fire")
root4 = Path("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/val/nofire")
fireValidationData = obtainData(root3) # Extracting the validation images containing wildfires
noFireValidationingData = obtainData(root4) # Extracting the validation images containing no wildfires
validationData = np.array(fireValidationData + noFireValidationingData) # Combining folders to complete the validation data


# Extracting Test Data
root4 = Path("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/test/fire")
root5 = Path("/home/marvin/WildFireClassification/Data/the_wildfire_dataset_2n_version/test/nofire")
fireTestData = obtainData(root4) # Extracting the test images containing wildfires
noFireTestData = obtainData(root5) # Extracting the test images containing no wildfires
testData = np.array(fireTestData + noFireTestData) # Combining both folders of images to complete the training data

# reshape data so that we can input it into CNN as single channel image
trainingData = trainingData.reshape(trainingData.shape[0], 64, 64, 3)
validationData = validationData.reshape(validationData.shape[0], 64, 64, 3)
testData = testData.reshape(testData.shape[0], 64, 64, 3)

# Labels
trainingLabels = np.array([1]*len(fireTrainingData)+[0]*len(noFireTrainingData)) # Training labels 
validationLabels = np.array([1]*len(fireValidationData)+[0]*len(noFireValidationingData)) # Training labels 
testLabels = np.array([1]*len(fireTestData)+[0]*len(noFireTestData)) # Training labels

# Normalizing the training, validation and test data
xTrain = trainingData.astype(np.float32) / 255.0
xValid = validationData.astype(np.float32) / 255.0
xTest = testData.astype(np.float32) / 255.0

# Convolution Neural Network Structure 
model = Sequential()

# Data: RGB Images 64x64
model.add(Input(shape=(64,64,3))) # Three Channels: R, G, B

# Convolution Layers: 3-layer strcuture

# First convolution block learns low-level features
model.add(Conv2D(32, (3,3), padding='same'))

# Applyinng Normalization to CNN Layers
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

# Converting into values 
model.add(Flatten()) # Flatten the feature maps into a 1D vector

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid')) # Sigmoid: Binary Classfication 

# compile and train model CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xTrain, trainingLabels, epochs=15, batch_size=32, validation_data=(xValid, validationLabels))

# Convolutional Neural Network Preformance Block

print("Convolutional Nerual Network Results")
# predictions
y_pred = model.predict(xTest)
y_pred = (y_pred > 0.5).astype(int)

# evaluation
loss, acc = model.evaluate(xTest, testLabels) # Obtaining kiss and accuracy 