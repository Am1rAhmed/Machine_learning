import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#diabetes = datasets.load_diabetes()
diabetes_x = np.array([[1], [2], [3]])
diabetes_x_train = diabetes_x
diabetes_x_test = diabetes_x

diabetes_y_train = np.array([3, 2, 4])
diabetes_y_test = np.array([3, 2, 4])

model = linear_model.LinearRegression()

model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_x_test)

print("mean squared error is :", mean_squared_error(
    diabetes_y_test, diabetes_y_predicted))

print("weights", model.coef_)
print("Intercept", model.intercept_)
# for plotting line:
plt.plot(diabetes_x_test, diabetes_y_predicted)

# for plotting graph:
plt.scatter(diabetes_x_test, diabetes_y_test)
plt.show()
# mean squared error is : 3035.060115291269
# weights [941.43097333]
# Intercept 153.39713623331644
