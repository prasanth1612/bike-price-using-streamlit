# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:26:31 2023

@author: ADMIN
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
import pickle
import json

dataset=pd.read_csv('bike.csv')
dataset.describe()

dataset.plot(x='mileage',y='sell price',style="o")

x=dataset[["mileage","age"]]
y=dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=8) 

regressor=LinearRegression() 
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
df=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
df

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(regressor.intercept_)
print(regressor.coef_)
print(regressor.score(x_train,y_train))
regressor.predict([[57000,5]])

pickle.dump(regressor, open('reg.pkl','wb'))

model1 = pickle.load(open('reg.pkl','rb'))
print(model1.predict([[185.9,102]]))