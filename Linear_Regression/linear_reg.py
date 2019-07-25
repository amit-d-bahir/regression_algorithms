# *** SIMPLE LINEAR REGRESSION ***


# ***Data Preprocessing***
# ***Importing libraries***
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ***Importing the dataset*** 
df = pd.read_csv('Salary_Data.csv')

    # Feature Matrix
X = df.iloc[:, :-1].values

    # Dependent variable vector
y = df.iloc[:, -1:].values


# ***Spiltting the dataset into Training set and Test set***
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=0)

"""
#  *** Feature Scaling ***
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# ***Fitting Simple Linear Regression to Training Set***
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ***Predicting Test set Results***
y_pred = regressor.predict(X_test)

# ***Visualizing the Training Set Results
    # This is for Training Set 
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "powderblue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $")
plt.show()

    # This is for Test Set 
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "powderblue")
    # We have not changed the train here -> Our regressor is fit on training set so it will predict according to it
    # no matter what, Changing it to test will just build some new predictions of the test set observation points
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $")
plt.show()