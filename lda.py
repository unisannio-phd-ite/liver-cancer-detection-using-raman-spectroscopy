# -*- coding: utf-8 -*-
"""LDA.ipynb
#  Linear Descriminant Analaysis on  Raman spectroscopy Data
1. This program will perform LDA prediction on Raman spectroscopy data
2. The input Raman spectroscopy data should be preprocessed (i.e., background removal, baseline correction, normalization, and outlier removal)
2. The Raman spectroscopy data can either be in a text, csv, or xml format
3. The Raman spectrosocpy data has been transposed with the last column being the classes while the rest of the columns are variables
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# %matplotlib inline
import math
np.set_printoptions(precision=4)

# evaluate a lda model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict,
                                     StratifiedKFold, learning_curve)

from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
from collections import Counter

import seaborn as sns
sns.set()
sns.set(style = 'white' , context = 'notebook', palette = 'deep')
warnings.filterwarnings('ignore', category = DeprecationWarning)
# %matplotlib inline

"""# Loading preprocessed and transposed Raman spectroscopy data
1. The Raman spectroscopy data contains 180 non tumor and 181 tumor spectra and in total are 361 spectra.
"""

# Loading Vector normalized cells data (i.e., tumor and non tumor cells)
# The LDA model is build using know Cell spectra acquired with Raman spectroscopy
dp=pd.read_csv('./data/1-40NTNC_1_40TNC_Nu_FP.csv')
dp.head()

# Definating variable and their classes
X= dp.loc[:, 'C1':'C364'].values#X= variable
y= dp['target'].values#y= target classes

#Estimating total number of NTNC and TNC Cell spectral
import string
unique, counts = np.unique(y, return_counts=True)
dict(zip(unique, counts))

"""# Building the model by spliting the dataset into 80% train and 20% reserved as test-set

"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#Standardize data train and test dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

feature_importances = np.zeros(X_train.shape[1])

# Cross validate model with Kfold stratified cross val
from sklearn.model_selection import StratifiedKFold, KFold
K_fold = StratifiedKFold(n_splits=5)

# Linear Discriminant Analysis
LDA_Model= LinearDiscriminantAnalysis()

scores = cross_val_score(LDA_Model, X_train, y_train, cv = K_fold, n_jobs = 4, scoring = 'accuracy')

print(scores)
round(np.mean(scores)*100, 2)

"""# Testing LDA model accuracy on 20%  test-dataset"""

LDA_Model=LDA_Model.fit(X_train, y_train)
y_pred1=LDA_Model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,y_pred1))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred1)))
print(classification_report(y_test,y_pred1))

"""# The learning curve plot
1. the learning curve plot algorithm was adopted from scikitlearn, for more information see: https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html
"""

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

"""# The learning curve plot"""

# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=15)
title='LDA learning curve'
estimator = LinearDiscriminantAnalysis()
plot_learning_curve(estimator, title, X, y, ylim=(0.4, 1.01), cv=cv, n_jobs=1)

plt.title("LDA learning curve",fontsize = 20)
plt.ylabel('Score', fontsize=25)
plt.xlabel('Training examples', fontsize=25)

plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
plt.legend(loc="lower right",fontsize=15)
xlim=(0, 60)
plt.show()

"""# Hyperparameter Tuning:
1. LDA hyperparameters are tuned using GridSerach algorithm
2. For more information of GridSearch (see:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
3. For our studies, we found these tol= [0.001,0.05,0.3] to work fine. However, one can still change the tol manually to improve the GridSearch optimization.
"""

# Linear Discriminant Analysis - Parameter Tuning
LDA = LinearDiscriminantAnalysis()

## Search grid for optimal parameters
lda_param_grid = {"solver" : ["svd"],"tol" : [0.001,0.05,0.3]}


gsLDA = GridSearchCV(LDA, param_grid = lda_param_grid, cv=K_fold,
                     scoring="accuracy", n_jobs= 4, verbose = 1)

gsLDA.fit(X_train,y_train)
LDA_best = gsLDA.best_estimator_

# Best score
gsLDA.best_score_
y_pred=LDA_best.predict(X_test)

"""# Testing LDA  hyperparameter tuning using GridSearch accuracy on 20% test-dataset"""

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))

"""# The learning curve plot
1. the learning curve plot algorithm was adopted from scikitlearn, for more information see: https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html
"""

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    x1 = np.linspace(0, 10, 8, endpoint=True) produces
        8 evenly spaced points in the range 0 to 10
    """


    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

import matplotlib as mpl
plt.figure(figsize=(20, 30))
title='b'

cv1 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=15)

estimator0 = gsLDA.best_estimator_
plot_learning_curve(estimator0, title, X, y, ylim=(0.4, 1.01), cv=cv1, n_jobs=1)

plt.title("b",fontsize = 20)
plt.ylabel('Score', fontsize=25)
plt.xlabel('Training examples', fontsize=25)
plt.title
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
plt.legend(loc="lower right",fontsize=15)
xlim=(0, 260)

"""# Predicting the classes of blind Raman spectroscopy data using Hyperparameter tuned LDA  
1. The blind Raman spectroscopy data are preprocessed and transposed as described above
2. the Raman spectroscopy data are loaded as csv file format
3. blind Raman spectra contains 136 mix of tumor and non tumor sample
"""

d=pd.read_csv('./data/1-30MIX1_NU_FP.csv')
d.head()

df=d.loc[:,:].values
df1=sc.transform(df)

prd=LDA_best.predict(df1)
prd

import string
unique, counts = np.unique(prd, return_counts=True)
dict(zip(unique, counts))
# NTNC=non tumor non cultivated
#TNC=tumor non cultivated

d1=pd.read_csv('./data/1_20MIX2_NU_FP.csv')
d1.head()

dft=d1.loc[:,:].values
dft1=sc.transform(dft)

prd1=LDA_best.predict(dft1)
prd1

import string
unique, counts = np.unique(prd1, return_counts=True)
dict(zip(unique, counts))
# NTNC=non tumor non cultivated
#TNC=tumor non cultivated

