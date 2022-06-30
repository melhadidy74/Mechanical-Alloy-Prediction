#Import Necessary Libraries
import numpy as np                   #Numpy library to perform math operations on data
import pandas as pd                  #Pandas library to deal with datasets different types such as .csv
import matplotlib.pyplot as plt      #Matplotlib library to plot & visulaize our data
import seaborn as sns                #Seaborn library is another library used in visulazation
from sklearn import preprocessing    #Preprocessing for data before applying ML model
from sklearn.preprocessing import StandardScaler #Standardscaler to scale our data
from sklearn.model_selection import train_test_split #Train_test_split to divide our data to train and test splits
from sklearn.ensemble import RandomForestRegressor #Import random forest regression to do regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error #our model's evaluation matrices
import pickle
#Importing our dataset
df = pd.read_csv("C:\ML Model Deployment\MatNavi Mechanical properties of low-alloy steels.csv")
#Dropping Alloy code as it won't be used in our model
df.drop('Alloy code', axis = 1, inplace = True)

#Drop row with index = 626
df.drop(index = 626, axis = 1, inplace = True)

#Let's split our dataframe into features and targets
x = df.drop(df.iloc[:,-4:], axis = 1)
y = df.iloc[:,-4:]
#Let's Split our data into train/test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.3, random_state =40)
#Fitting and predicting our model
RF = RandomForestRegressor()              #Assign the regression module to RF
RF = RF.fit(x_train, y_train)             #Fit our train data to the model

filename = 'model.pkl'
pickle.dump(RF, open(filename, 'wb'))