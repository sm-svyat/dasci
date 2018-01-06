#   Centering transformation is basically reducing the Mean value of samples from all observations. So, the observations
#   will have a mean value of Zero after this transformation. Scaling transformation is dividing value of predictor for
#   each observation by standard deviation of all samples. This will cause the transformed values to have a standard
#   deviation of One.

#   When & Why we need Centering & Scaling (Standardization):

#   1) Standardization is recommended when regression models are being built. When there are predictors with different units
#   and ranges, the final model will have coefficients which are very small for some predictors and it makes it difficult
#   to interpret

#   2) Centering & Scaling will improve the numerical stability of some models(i.e PLS)

#   3) Many predictive modeling techniques use the predictor variance as an important factor for assigning importance
#   to each predictor(PLS,…). In this situation, since variables with larger units usually have higher variance compare
#   to predictors which have smaller units, the models will favor variables with larger units. The Centering & Scaling
#   transformation ensures that unit differences don’t impact predictor selection and final model.

#   There is a field in this dataset named AirTime which shows flight time in Minutes. We show the histograms, standard
#   deviation and mean of this field before and after Centering & Scaling transformations:

#Airline Data Example
#Centering & Scaling Flight Time Using Python
#You can download dataset from http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236

import pandas as pd
import matplotlib.pylab as plt
from sklearn import preprocessing

#First we import the data
data = pd.read_csv('data/On_Time_On_Time_Performance_2015_1.csv')
print(data.index)
#Replace Missing Values with zero
data['AIR_TIME'].fillna(0,inplace=True)

#The next line uses scale method from scikit-learn to transform the distribution
airTime = preprocessing.scale(data['AIR_TIME'])

#We draw the histograms side by side
figure = plt.figure()
ax1 = figure.add_subplot(131)
plt.hist(data['AIR_TIME'], bins=100, facecolor='red',alpha=0.75)
plt.xlabel("AirTime(Minutes)")
plt.ylabel("Frequency")
plt.title("Original Flight Time Histogram")
ax1.text(300,100000,"Mean: {0:.2f} \n Std: {1:.2f}".format(data['AIR_TIME'].mean(),data['AIR_TIME'].std()))

ax2 = figure.add_subplot(133)
plt.hist(airTime,bins=100, facecolor='blue',alpha=0.75)
plt.xlabel("AirTime - Transformed")
plt.title("Transformed AirTime Histogram")
ax2.text(2, 100000,"Mean: {0:.2f} \n Std: {1:.2f}".format(airTime.mean(),airTime.std()))
plt.show()