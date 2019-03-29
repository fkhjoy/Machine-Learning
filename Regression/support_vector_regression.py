#importing libraries
import numpy as np
import pylab as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

#feature scaling
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
scale_y = StandardScaler()
X = scale_x.fit_transform(X)
y = scale_y.fit_transform(y)

#fitting svr
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, y)

#predicting
y_pred = scale_y.inverse_transform(regressor.predict(scale_x.transform(np.array([[6.5]]))))

#visualization of svr resutls
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Support Vector Regression (Truth)')
plt.xlabel('Positoin Level')
plt.ylabel('Salary')
plt.show()
