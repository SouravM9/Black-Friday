import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#load data and remove NANs

data = pd.read_csv("BlackFriday.csv")
#print data.info()
#print data.describe()

data.fillna(value=0, inplace=True)
data["Product_Category_2"] = data["Product_Category_2"].astype(int)
data["Product_Category_3"] = data["Product_Category_3"].astype(int)

data.drop(columns = ["User_ID","Product_ID"],inplace=True)


#print data.info()

data["Stay_In_Current_City_Years"].replace('4+', '4', inplace=True)
data["Stay_In_Current_City_Years"] = data["Stay_In_Current_City_Years"].astype(int)
data["City_Category"].replace('A', 2, inplace=True)
data["City_Category"].replace('B', 1, inplace=True)
data["City_Category"].replace('C', 0, inplace=True)
#print data["City_Category"].unique()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


#Preprocessing
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [0, 1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

#Fitting into model
'''
#Multiple  Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print regressor.score(X_test, y_test)

#Errors
print mean_squared_error(y_test, y_pred)
print r2_score(y_test, y_pred)

#SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print regressor.score(X_test, y_test)

#Errors
print mean_squared_error(y_test, y_pred)
print r2_score(y_test, y_pred)


#Decision Tree
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print regressor.score(X_test, y_test)

#Errors
print mean_squared_error(y_test, y_pred)
print r2_score(y_test, y_pred)
'''

#Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print regressor.score(X_test, y_test)

#Errors
print mean_squared_error(y_test, y_pred)
print r2_score(y_test, y_pred)