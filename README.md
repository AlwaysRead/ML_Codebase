# Machine Learning Codebase

A comprehensive collection of machine learning implementations covering fundamental algorithms and techniques. Each program is provided as a Jupyter notebook with complete code, explanations, and visualizations.

## Overview

This repository contains 10 core machine learning programs that demonstrate essential concepts from data exploration to advanced algorithms. The implementations are designed for educational purposes and practical understanding of machine learning fundamentals.

## Programs

### Data Analysis and Visualization

**Program 1: Exploratory Data Analysis**  
Questions: Develop a program to create histograms for all numerical features and analyze the distribution of each feature. Generate box plots for all numerical features and identify any outliers. Use California Housing dataset.
- Create histograms for numerical features and analyze distributions
- Generate box plots to identify outliers
- Dataset: California Housing
- File: [`Pg1.ipynb`](./Pg1.ipynb)

**Program 2: Correlation Analysis**  
Questions: Develop a program to Compute the correlation matrix to understand the relationships between pairs of features. Visualize the correlation matrix using a heatmap to know which variables have strong positive/negative correlations. Create a pair plot to visualize pairwise relationships between features. Use California Housing dataset.
- Compute correlation matrices between feature pairs
- Visualize correlations using heatmaps
- Create pair plots for comprehensive relationship analysis
- Dataset: California Housing
- File: [`Pg2.ipynb`](./Pg2.ipynb)

### Dimensionality Reduction

**Program 3: Principal Component Analysis (PCA)**  
Questions: Develop a program to implement Principal Component Analysis (PCA) for reducing the dimensionality of the Iris dataset from 4 features to 2.
- Implement PCA for dimensionality reduction
- Reduce Iris dataset from 4 features to 2 dimensions
- Dataset: Iris
- File: [`Pg3.ipynb`](./Pg3.ipynb)

### Machine Learning Algorithms

**Program 4: Find-S Algorithm**  
Questions: For a given set of training data examples stored in a .CSV file, implement and demonstrate the Find-S algorithm to output a description of the set of all hypotheses consistent with the training examples.
- Implement the Find-S concept learning algorithm
- Generate hypotheses consistent with training examples
- Input: CSV training data
- File: [`Pg4.ipynb`](./Pg4.ipynb)

**Program 5: k-Nearest Neighbors (k-NN)**  
Questions: Develop a program to implement k-Nearest Neighbour algorithm to classify the randomly generated 100 values of x in the range of [0,1]. Perform the following based on dataset generated. a) Label the first 50 points {x1,……,x50} as follows: if (xi ≤ 0.5), then xi ∊ Class1, else xi ∊ Class1 b) Classify the remaining points, x51,……,x100 using KNN. Perform this for k=1,2,3,4,5,20,30
- Classify randomly generated data points using k-NN
- Test with k values: 1, 2, 3, 4, 5, 20, 30
- Generate 100 random values in range [0,1]
- Binary classification based on threshold 0.5
- File: [`Pg5.ipynb`](./Pg5.ipynb)

**Program 6: Locally Weighted Regression**  
Questions: Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.
- Implement non-parametric locally weighted regression
- Fit data points with adaptive local models
- Includes visualization and performance analysis
- File: [`Pg6.ipynb`](./Pg6.ipynb)

**Program 7: Regression Analysis**  
Questions: Develop a program to demonstrate the working of Linear Regression and Polynomial Regression. Use Boston Housing Dataset for Linear Regression and Auto MPG Dataset (for vehicle fuel efficiency prediction) for Polynomial Regression.
- Linear Regression implementation using Boston Housing dataset
- Polynomial Regression implementation using Auto MPG dataset
- Compare performance between linear and polynomial approaches
- File: [`Pg7.ipynb`](./Pg7.ipynb)

**Program 8: Decision Tree Classification**  
Questions: Develop a program to demonstrate the working of the decision tree algorithm. Use Breast Cancer Data set for building the decision tree and apply this knowledge to classify a new sample.
- Build decision tree classifier for breast cancer detection
- Apply trained model to classify new samples
- Dataset: Breast Cancer
- File: [`Pg8.ipynb`](./Pg8.ipynb)

**Program 9: Naive Bayes Classification**  
Questions: Develop a program to implement the Naive Bayesian classifier considering Olivetti Face Data set for training. Compute the accuracy of the classifier, considering a few test data sets.
- Implement Naive Bayesian classifier for face recognition
- Calculate classification accuracy on test data
- Dataset: Olivetti Face
- File: [`Pg9.ipynb`](./Pg9.ipynb)

**Program 10: k-Means Clustering**  
Questions: Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set and visualize the clustering result.
- Implement k-means clustering algorithm
- Visualize clustering results and centroids
- Dataset: Wisconsin Breast Cancer
- File: [`Pg10.ipynb`](./Pg10.ipynb)

## Datasets

The following datasets are used across the programs:

- California Housing Dataset - Housing prices and features
- Iris Dataset - Flower classification with 4 features
- Boston Housing Dataset - Real estate price prediction
- Auto MPG Dataset - Vehicle fuel efficiency data
- Breast Cancer Dataset - Medical diagnosis classification
- Olivetti Face Dataset - Face recognition images
- Wisconsin Breast Cancer Dataset - Cancer diagnosis clustering

Most datasets are loaded directly through scikit-learn or downloaded automatically within the notebooks.

## Learning Objectives

Each program is designed to help understand:
- Data preprocessing and exploration techniques
- Statistical analysis and visualization methods
- Supervised learning algorithms
- Unsupervised learning approaches
- Model evaluation and performance metrics
- Algorithm implementation from scratch
