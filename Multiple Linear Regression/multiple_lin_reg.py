#  *** MULTIPLE LINEAR REGRESSION ***


# ***Data Preprocessing***
# ***Importing libraries***
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ***Importing the dataset*** 
df = pd.read_csv('50_Startups.csv')

    # Feature Matrix
X = df.iloc[:, :-1].values

    # Dependent variable vector
y = df.iloc[:, -1:].values


# ***Encoding Categorical Data for the independent variable***
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# *** Avoiding the DUMMY VARIABLE TRAP ***
X = X[:, 1:] # The python libraray takes care of this, no need to do it manually everytime

# ***Spiltting the dataset into Training set and Test set***
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# *** Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ***Predicting the Test Set results
y_pred = regressor.predict(X_test)

# *** Building the optimal model using BACKWARD ELIMINATION Method
    # Creating a column of 1s for B_o in our equation
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((40,1)).astype(int), values = X_train, axis = 1)

X_train_opt = X_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog= y_train, exog= X_train_opt).fit()
regressor_OLS.summary()
    # SL = 0.05 and eliminating those features which have p > SL

X_train_opt = X_train[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog= y_train, exog= X_train_opt).fit()
regressor_OLS.summary()

X_train_opt = X_train[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog= y_train, exog= X_train_opt).fit()
regressor_OLS.summary()

X_train_opt = X_train[:,[0,3,5]]
regressor_OLS = sm.OLS(endog= y_train, exog= X_train_opt).fit()
regressor_OLS.summary()
    # If we are gonna thoroughly follow backward elimination then we will eliminate 5th column
    # But we are gonna use other powerful metrics such as R-squared and Adj R-Squared to decide with
    # more certainity whether we need to keep it or not

y_train_opt = y_train
y_test_opt  = y_test

X_test = np.append(arr= np.ones((10,1)).astype(int), values= X_test, axis=1)
X_test_opt = X_test[:,[0,3,5]]