#simple linear regression using built in libraries
import pylab as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values #features
y = dataset.iloc[:, -1].values #targets

#splitting dataset into training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)

#visualization of training set results
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Experience vs Salary on train set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualization of test set results
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Experience vs Salary on test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()