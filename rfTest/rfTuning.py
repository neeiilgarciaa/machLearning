import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Insert dataset here
#df = pd.read_csv(r"/kaggle/input/heart/heart_2020_cleaned.csv")

columns = list(df.columns.values)
df.head()
df

# Convert the non-numeric to numerical
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()

features = columns
for feature in features:
    df[feature] = encoder.fit_transform(df[[feature]])

# define depenedent variable
Y = df['HeartDisease'].values
Y=Y.astype('int')

# define independent variables
X = df.drop(labels=['HeartDisease'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=(42))

# Implement RF & Importing Random Forest Classifier from the sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# Automate Hyperparameter Tuning using RandomSearchCV
# Input necessary parameters to test for Tuning

n_estimators = [100,500,1000,1500,2000] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # maximum number of levels allowed in each decision tree
min_samples_split = [3, 10, 15, 18, 22, 27, 30] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4, 5, 7, 9] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap}

# Set necessary parameters for number of folds and cross validation
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)

rf_random.fit(X_train, Y_train)

print ('Random grid: ', random_grid, '\n')
# Print the best parameters
print ('Best Parameters: ', rf_random.best_params_, ' \n')

# STOP the code here
## Use the best parameters from 
randmf = RandomForestRegressor(n_estimators = 100, min_samples_split = 2, min_samples_leaf= 1, max_features = 'sqrt', max_depth= 120, bootstrap=False) 
randmf.fit( X_train, Y_train)
prediction_test = randmf.predict(X_train)

# Provide metrics and measure performance
from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

# Feature Importance to weigh each variables affecting the independent variable
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

