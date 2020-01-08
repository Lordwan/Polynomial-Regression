import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[1], [2], [3], [4], [15]] #Years Worked
y_train = [[1000], [1250], [1500], [1750], [2000]] #Pay in Dollars

X_test = [[1.5], [0.5], [2], [5]] #Years worked
y_test = [[1000], [600], [1300], [1800]] #Pay in Dollars

regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))

quadratic_featurizer = PolynomialFeatures(degree=2)

X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Years worked vs Pay Recieved')
plt.xlabel('Years Worked')
plt.ylabel('Pay in dollars')
plt.axis([0, 5, 0, 2000])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print (X_train)
print (X_train_quadratic)
print (X_test)
print (X_test_quadratic)

