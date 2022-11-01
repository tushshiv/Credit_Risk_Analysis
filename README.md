# Credit Risk Analysis

## Overview
The purpose of this analysis is to understand how to use Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on Supervised Learning using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. 

To complete this analysis, we use different Machine Learning techniques to train and evaluate the data with unbalanced classes. The dataset from the LendingClub has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order balance out the classifications to allow for more meaningful predictions and improve the accuracy score, we needed to employ various Machine Learning algorithms to resample the data. These algorithms include RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN, BalancedRandomForestClassifier, and EasyEnsembleClassifier.

## Results
As mentioned above, We have used machine learning to sample the dataset using  Python libraries Scikit-learn and imbalanced -learn to evaluate the results and compare.

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk"

![pic_1](pic_1.jpg)

 75/25 method was used to split the data for training vs testing 51366 low risj and 246 high risk applications were categorized into training sets.
 
**Oversampling**
RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.
![pic_2](pic_2.jpg)

The "High Risk" precision rate was only 1% with the recall at 69% giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 100% and recall at 59%.

![pic_4](pic_4.jpg)


**SMOTE (Synthetic Minority Oversampling Technique) Model**, like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.



The balanced accuracy score improved slightly to 65.1%.
![smoteacc]

Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 61% giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 100% and an improved recall at 69%.

![smotecm]

![smoteclass]

Undersampling
ClusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.

![undersamplecount]

Balanced accuracy score was lower than the oversampling models at 54.5%.
![underacc]

The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
"Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.
![undercm]

![underclass]

Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk
Combination Sampling
SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk.

![SMOTEENNcount]

The balanced accuracy score improved to 64.5% when using a combined sampling model.
![SMOTEENNacc]

The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
"Low Risk" still showed a precision rate of 100% with the recall at 57%.
![SMOTEENNcm]

![SMOTEENNclass]

## Use Ensemble Classifiers to Predict Credit Risk
Compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.

![Balancedcount]

BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.

The balanced accuracy score increased to 78.9% for this model.
![balanceacc]

The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
"Low Risk" still had a precision rate of 100% with the recall at 87%.
The top feature by importance was "total_rec_prncp" at 7.9% of the total.


EasyEnsembleClassifier Model, a set of classifiers where individual decisions are combined to classify new examples.

The balanced accuracy score increased to 93.2% with this model.


The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
"Low Risk" still had a precision rate of 100% with the recall now at 94%.


## Summary
In reviewing all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.

**Ranking of models in descending order based on "High Risk" results:**

* EasyEnsembleClassifer: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
* BalancedRandomForestClassifer: 78.9% accuracy, 3% precision, 70% recall and 6% F1 Score
* SMOTE: 65.2% accuracy, 1% precision, 61% recall and 2% F1 Score
* SMOTEENN: 64.5% accuracy, 1% precision, 72% recall and 2% F1 Score
* RandomOverSampler: 64.0% accuracy, 1% precision, 66% recall and 2% F1 Score
* ClusterCentroids: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score

A side note that should be considered is that original dataset had 99% of the applications classified as "Low Risk" with only 1% of the data classified in the "High Risk" category. This may skew the results greatly as there is a risk that the Machine Learning algorithms are creating clusters drawing from too small of a dataset of actual "High Risk" applications. This margin of risk might not be something that banks would be comfortable accepting.
