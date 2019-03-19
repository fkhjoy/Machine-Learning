import pylab as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, 0].values #features
y = dataset.iloc[:, -1].values #targets

#considering the equation y = mx + c
#here y = target value, m = slope, x = feature value, c = intercept
m = 0
c = 0
learning_rate = .001
iteration = 50000
total_features = float(len(X))

precision = .00001

for i in range(iteration):
    
    up_m, up_c = m, c
    y_pred = m*X + c
    m_grad = (-2/total_features) * sum(X * (y - y_pred) ) # derivative with respect to m
    c_grad = (-2/total_features) * sum(y - y_pred) # derivative with respect to c
    
    m = up_m - learning_rate * m_grad
    c = up_c - learning_rate * c_grad
    step = (m - up_m + c - up_c)/2.0
    
    if(abs(step) <= precision):
        break
    
    
    
y_pred = m*X + c

plt.scatter(X, y, color = 'blue')
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')

#fitting line
plt.plot(X, y_pred, color = 'red')

plt.show()