# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:52:35 2023

@author: Neil Garcia
"""

import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\Neil Garcia\randomforest\cardio\data.csv")
#print(df.head())

# print(df.Age_Category.unique())

# drop irrelevant columns
df.drop(['Exercise', 'Green_Vegetables_Consumption', 'Fruit_Consumption', 'FriedPotato_Consumption'], inplace=True, axis=1)


#df['General_Health'].unique()


# convert alpha to numeric

binary_columns = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']

for column in binary_columns:
    df[column] = df[column].map({'Yes': 1, 'No': 2})

df.General_Health[df.General_Health == 'Excellent'] = 1
df.General_Health[df.General_Health == 'Very Good'] = 1.25
df.General_Health[df.General_Health == 'Good'] = 1.5
df.General_Health[df.General_Health == 'Fair'] = 1.75 
df.General_Health[df.General_Health == 'Poor'] = 2

df.Sex[df.Sex == 'Male'] = 2
df.Sex[df.Sex == 'Female'] = 1

df.Checkup[df.Checkup == 'Within the past year'] = 1
df.Checkup[df.Checkup == 'Within the past 2 years'] = 1.25
df.Checkup[df.Checkup == 'Within the past 5 years'] = 1.5
df.Checkup[df.Checkup == '5 or more years ago'] = 1.75 
df.Checkup[df.Checkup == 'Never'] = 2

df.Age_Category[df.Age_Category == '18-24'] = 1
df.Age_Category[df.Age_Category == '25-29'] = 1.08
df.Age_Category[df.Age_Category == '30-34'] = 1.16
df.Age_Category[df.Age_Category == '35-39'] = 1.25
df.Age_Category[df.Age_Category == '40-44'] = 1.33
df.Age_Category[df.Age_Category == '45-49'] = 1.41
df.Age_Category[df.Age_Category == '50-54'] = 1.5
df.Age_Category[df.Age_Category == '55-59'] = 1.58
df.Age_Category[df.Age_Category == '60-64'] = 1.66
df.Age_Category[df.Age_Category == '65-69'] = 1.75
df.Age_Category[df.Age_Category == '70-74'] = 1.83
df.Age_Category[df.Age_Category == '75-79'] = 1.91
df.Age_Category[df.Age_Category == '80+'] = 2
df.Diabetes[df.Diabetes == 'Yes'] = 1
df.Diabetes[df.Diabetes == 'No'] = 2
df.Diabetes[df.Diabetes == 'No, pre-diabetes or borderline diabetes'] = 2
df.Diabetes[df.Diabetes == 'Yes, but female told only during pregnancy'] = 2

# define depenedent variable
Y = df['General_Health'].values
Y=Y.astype('int')

# define independent variables
X = df.drop(labels=['General_Health'], axis=1)


# split data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=(40))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=(10))

model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)


from sklearn import metrics
from sklearn.metrics import classification_report

print("\nAccuracy = ", metrics.accuracy_score(Y_test, prediction_test))
print(classification_report(Y_test, prediction_test)) #y_test and predicted test - x_test


feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp, "\n")

