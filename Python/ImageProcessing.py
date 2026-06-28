import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def imageProcessing(path:str):
    # Converting image to GrayScale
      # Obtaining 1-dimensional arrays for each image 
    root = Path(path)
    data = []
    for image_path in root.rglob("*.jpg"):

        img = cv2.imread(str(image_path), 0) # Gresy Scale Image
        # Skip Invalid Images 
        if img is None:
            continue

        img = cv2.resize(img, (64,64)) # Uniform Dimensions for all data 

        features = img.flatten() # Turning Image into 1-D array for the logistic regresssion 

        data.append(features) # Appending all the features of the training data 
    return list(data) # Return Feature List 

def main():
    models = ["LogReg", "Naive Bayes", "CNN"]

    accuracy = [0.71, 0.69, 0.82]
    precision = [0.65, 0.57, 0.73]
    recall = [0.53, 0.78, 0.82]
    f1 = [0.59, 0.66, 0.78]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10,6))
    plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
    plt.bar(x - 0.5*width, precision, width, label='Precision')
    plt.bar(x + 0.5*width, recall, width, label='Recall')
    plt.bar(x + 1.5*width, f1, width, label='F1')

    plt.xticks(x, models)
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()