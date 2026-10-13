import pandas as pd 
import numpy as np

# Splits the dataset into training and testing sets.
from sklearn.model_selection import train_test_split
# Finds the best hyperparameters using cross-validation.
from sklearn.model_selection import GridSearchCV

# Scales features to a specified range, usually 0 to 1.
from sklearn.preprocessing import MinMaxScaler

# Classifies data based on the nearest neighbors.
from sklearn.neighbors import KNeighborsClassifier

# Calculates the proportion of correct predictions.
from sklearn.metrics import accuracy_score

# Shows correct and incorrect predictions for each class.
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt 
import seaborn as sns 


data = pd.read_csv("titanic.csv")