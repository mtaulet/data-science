# data-science
This repository houses personal projects based on data science and machine learning

# Credit Card Fraud
Data: https://www.kaggle.com/mlg-ulb/creditcardfraud

This project implements Naive Bayes and Isolation Forest algorithms to predict fraud in credit card transactions. The goal is to compare the two models in an anomaly detection problem with an imbalanced dataset.

# Adopted Users
Given a dataset with details about the creation of 12,000 online accounts for an online product, and a second dataset listing the days in which each user logged in to the product, the goal is to determine the features that make a user more likely to be an "adopted" user in the future. An "adopted" user is defined as a user that logs in to the product at least three times in a seven day period.

# Eigenfaces
Eigenfaces is the name given to a set of eigenvectors (principal components) when they are used in the computer vision problem of human face recognition. These eigenvectors/principal components can be derived from the covariance matrix of 𝑋, where 𝑋 is a 𝑑  x  𝑛  matrix such that each column of 𝑋 represents an image. (One of the ways of interpreting PCA is projecting data onto the  𝑘  leading eigenvectors of the sample covariance matrix).

The goal of this project is to extract the top eigenvectors of a dataset of celebrity images and reconstruct the images as faithfully to the originals as possible by projecting the data onto the top eigenvectors to reduce the size of the data as efficiently as possible.

# Predicting parkinson's
The goal of this project is to apply PCA to the Parkinson's Disease Classification dataset and then perform classification using the k-nearest neighbors algorithm.

# Expectation-Maximization Algorithm
The expectation-maximization algorithm is an iterative method to find maximum likelihood (ML) estimates of parameters in statistical models. The E-M algorithm alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step. This alternation repeats until convergence.

The goal of this project is to perform E-M first on a sample Gaussian mixture model (GMM). Doing this will prove that the algorithm works, since the parameters of the model are already known.
