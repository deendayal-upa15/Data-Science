# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 19:55:16 2022

@author: Deendayal
"""

"""
CLASSIFYING PERSONAL INCOME


"""

#importing modules

import pandas as pd    #for Dataframes
import numpy as np     # for numerical operations
import seaborn as sns # data visualization
from sklearn.model_selection import train_test_split   #partition the data
from sklearn.linear_model import LogisticRegression    #for logistic regression
from sklearn.metrics import accuracy_score, confusion_matrix   #performance metricx

#importing data and making a copy

income_data = pd.read_csv('incomeData.csv')
data = income_data.copy()

"""
steps for data analysis
-- Getting to know the data
-- Data preprocessing (missing values)
-- Cross tables and data visualization
"""


"""Getting to know data"""

print(data.info()) #for checking data type
#data.isnull()       #for checking missing values
print('Data columns with null values:\n', data.isnull().sum())

summary_num = data.describe()   #numerical variables
print(summary_num)

summary_cate = data.describe(include = "O") #gives for categories count, unique, top, freq
print(summary_cate)

#frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

#we found some '?' instead of nan

#reading again by including "na_values[' ?']" to consider ' ?' as nan
data = pd.read_csv('incomeData.csv', na_values=[" ?"])


""" Data Pre-processing """
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis=1 : to consider at least one column value is missing

"""
Points noted:
    -- Missing values in JobType    = 1809
    -- Missing values in occupation = 1816
    -- There are 1809 rows to specific columns
    -- 1816-1809=7 are unfilled, beacause Jobtype is never worked
"""

data2 = data.dropna(axis=0)

#relationship between independent variables
correlation = data2.corr()


""" Cross table and Data visualization """

#extracting the column names
data2.columns

#gender proportional table
gender = pd.crosstab(index = data2["gender"], columns = 'count', normalize = True)
print(gender)

#gender vs salary status
gender_salstat = pd.crosstab(index = data2["gender"], columns = data2['SalStat'], margins = True, normalize = 'index')
print(gender_salstat)

#frequency distribution of 'salary status'
SalStat = sns.countplot(data2['SalStat'])
# 75% of poeple's salary status <= 50000
# 255 ..... > 50000

""" Histogram of Age """
sns.distplot(data2['age'], bins = 10, kde = False)
# people with age 20-25 age are high frequency

""" Box plot - Age vs Salary status """
sns.boxplot('SalStat', 'age', data = data2)
data2.groupby('SalStat')['age'].median()
# people with 35-50 age are more likely to earn > 50000
# people with 25-35 .... <= 50000

""" go through every factor to check if that can contribute in classification or not
only contributing factores are important"""



""" LOGISTIC REGRESSION """


#reindexing the salary status names to 0,1 beacause ML algo can not work with categorical data directly

data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first=True) #makes dummy data which has only 0, 1 for every category

# Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

# Separating the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

#storing the output values in y
y=new_data['SalStat'].values
print(y)

#storing the values from input features
x=new_data[features].values
print(x)

#splitting the data into train and test
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state=0)

#make an intance of the model
logistic = LogisticRegression()

# fitting the values for x and y
logistic.fit(train_x, train_y)
logistic.coef_  #coeficient value of all features
logistic.intercept_

#prediction from test data
prediction = logistic.predict(test_x)
print(prediction)


""" Confusion matrix """
#checking how much values are predicted wrong or right

confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix) 

""" prediction -->
actual |
      \/
      """

# calculating accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

#printing missclassified values from presiction
print('misclassified samples: %d' %(test_y != prediction).sum())


""" KNN """

#importing KNN library and plotting 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors= 5)

#fitting the vaues for x and y
KNN_classifier.fit(train_x, train_y)
 
# predicting the test values with model
prediction = KNN_classifier.predict(test_x)

#performance metrix check
confusion_matrix = confusion_matrix(test_y, prediction)
print("\t','Predicted values")
print('original values','\n',confusion_matrix)

# calculating accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

#printing missclassified values from presiction
print('misclassified samples: %d' %(test_y != prediction).sum())


""" effect of K value on classifier """

Misclassified_sample = []
#calculating error for K values between 1 to 20
for i in range (1 , 20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append(test_y != pred_i).sum()
    
print(Misclassified_sample)

""" End of Script """
