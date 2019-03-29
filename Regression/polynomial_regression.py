#importing libraries
import pylab as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#fitting linear regression
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 2)
X_poly = poly_regressor.fit_transform(X)
lin_regressor2 = LinearRegression()
lin_regressor2.fit(X_poly, y)

#visualization of linear regression resutls
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regressor.predict(X))
plt.title('Linear Regression (Truth)')
plt.xlabel('Positoin Level')
plt.ylabel('Salary')
plt.show()

#visualization of polynomial linear regression
plt.scatter(X, y, color =  'red')
plt.plot(X, lin_regressor2.predict(poly_regressor.fit_transform(X)))
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting with linear regression
lin_regressor.predict(6.5)

#predicting with polynomial regression
lin_regressor2.predict(poly_regressor.fit_transform(6.5))
