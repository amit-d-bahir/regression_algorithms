# ***POLYNOMIAL LINEAR REGRESSION ***

# ***Data Preprocessing***
# ***Importing libraries***
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ***Importing the dataset*** 
df = pd.read_csv('Position_Salaries.csv')

# The position values are already encoded in the level columns from 1-10
# So there is no need for encoding
    # Feature Matrix
X = df.iloc[:, 1:2].values
    # 1:2 -> so as to make it a matrix and not a vector
    
    # Dependent variable vector
y = df.iloc[:, -1:].values

# We have not done the splitting as we have very little data and we want to learn the maximum info
# and we want to make accurate prediction
# We don't do feature scaling too

# *** Fitting Linear Regression to the model***
    # We do this so, we can compare the result with the polynomial regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# ***Fitting Polynomial Regression to the dataset***
    # First we'll create the matrix consisting of polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
X_poly = poly_reg.fit_transform(X)
    # This also adds the constant 1 column at index 0, which we require for B_o in our equation
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# ***Visualizing the Linear Regression Result***
plt.scatter(X, y, color = "blue")
plt.plot(X, lin_reg.predict(X), color = "magenta")
plt.title("Linear Regression(Simple)")
plt.xlabel("Position Level")
plt.ylabel("Salary in $")
plt.show()


#  ***Visulaizing the Polynomial Regression Result***

"""plt.scatter(X, y, color = "blue") # Keeping the real observation points
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "magenta") # for degree=2
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary in $")
plt.show()"""

"""plt.scatter(X, y, color = "blue") # Keeping the real observation points
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "magenta") # for degree=3
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary in $")
plt.show()"""

plt.scatter(X, y, color = "blue") # Keeping the real observation points
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "magenta") # for degree=4
    # We have done this so we can change only X when we want to check for any new feature matrix
    # or we want to change the degree
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary in $")
plt.show()

"""# Now to make the curve smooth, we add a number of small values between the levels 
X_grid = np.arange(min(X), max(X), 0.1)
# Reshaping it
X_grid = X_grid.reshape((len(X_grid), 1))
#  ***Visulaizing the Polynomial Regression Result***
plt.scatter(X, y, color = "blue") # Keeping the real observation points
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "magenta") # for degree=4
    # We have done this so we can change only X when we want to check for any new feature matrix
    # or we want to change the degree
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary in $")
plt.show()"""

# ***Predicting a new result with Linear Regression***
arr = np.array([[6.5]])
arr.reshape((-1,1))
lin_reg.predict(arr)

# ***Predicting a new result with Polynomial Regression***
lin_reg_2.predict(poly_reg.fit_transform(arr))

