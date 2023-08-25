# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:12:54 2023

@author: Neil Garcia
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\Neil Garcia\randomforest\heart\heart_2020_cleaned.csv")
columns = list(df.columns.values)

df.head()


from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()

feature = columns
for feature in features:
    df[feature] = encoder.fit_transform(df[[feature]])


# define depenedent variable
Y = df['HeartDisease'].values
Y=Y.astype('int')

# define independent variables
X = df.drop(labels=['HeartDisease'], axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=(20))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=(100), max_depth=None, min_samples_leaf=5, min_samples_split=5)

model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)
