## Problem
- How do classical machine learning models compare to a Deep Learning model for Wildfire Detection using images?
- Can machine learning models detect wildfires with minimal true-negatives

## Motviations 
- Wildfires are extremely common within California. In 2026 more than 50,000 acres of land have been destroyed by wildfires.
- Accurate & precise wildfire detection could prevent wildfires from growing with early detection. 
- Identifying what models work best will save the lives & homes.


## Exploration
- Analysing a popular approach to wildfire detection to determine it's advantages and drawbacks. 
- Further analysis will be conducted between wildfire detection models utilizing enviromental data and images to determine each model's pro suitability. 

## Dataset  

**Dataset split:**
<br>

- Training: 70% 
- Validation: 15%
- Testing: 15%
<br>

**Data:** 
- RGB Images: 2700

**Binary classfication:** 
- Fire
- No Fire

**Dataset:** https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset

**Citation:**
El-Madafri I, Peña M, Olmedo-Torre N. The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach. Forests. 2023; 14(9):1697.
https://doi.org/10.3390/f14091697

## Approach 

**Classical models: Logistic regression & Gaussian naive bayes**
- Process images
    - Flatten them into 1D vectors and normalize them.
- Model training
    - Collect important metrics for Wildfire detection:
        - Recall 
        - Precision 
        - F1
        - Accuracy 
- Justification:
    - Classical models are easier to train, work well on small to moderatly sized datasets. 
    - Most importantly they're easier and faster to train compared to deep learning models. 

**Deep learning: Convolutional neural network (CNN)**
- Process images:
    - Collect RGB images and normalize.
- Design CNN structure
    - 3-layer structure 
    - RELU activation 
    - Sigmoid
- Model training:
    - Collect important metrics for Wildfire detection:
        - Recall 
        - Precision 
        - F1
        - Accuracy 

## Results: 

**Classical Models:**
- Logistic Regressiong achieved the highest classical model preformance with 71% accuracy and a 0.71 weighted F1-score, while Gaussian Naive Bayes achieved 69% accuracy and a 0.69 weighted F1-score.
- Although Gaussian Naive Bayes produced a higher wildfire precision (0.82), both classical models showed lower recall and F1-scores for wildfire detection than the CNN.

**Deep Learning:**
- The CNN significantly outperformed the classical models, achieving 85% accuracy and a 0.85 weighted F1-score, demonstrating superior overall classification performance.
- The model maintained strong performance across both classes, achieving F1-scores of 0.88 for No Fire and 0.80 for Fire, indicating effective and balanced wildfire detection.

