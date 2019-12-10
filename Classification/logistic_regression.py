#importing libraries
import  numpy as np
import pandas as pd

#importing datasets

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,  [2,3]].values
y = dataset.iloc[:, -1].values

#splitting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,  random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#fitting logistic Regression to the 
