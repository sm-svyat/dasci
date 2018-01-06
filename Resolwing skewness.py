#   The fundamental assumption in many predictive models is that the predictors have normal distributions.
#   Normal distribution is un-skewed. An un-skewed distribution is the one which is roughly symmetric.
#   It means the probability of falling in the right side of mean is equal to probability of falling on left side of mean.
#   The statistics for sample skewness is being calculated using below formula:

#   $latex Skewness = \frac{\sum(x_{i}-\overline{x})^{3}}{(n-1)\nu^{3/2}}&s=3$
#   $latex where\quad \nu = \frac{\sum(x_{i}-\overline{x})^{2}}{(n-1)}&s=3$

#Airline Data Example
#Calculating Skewness Statistic For Flight Time Using Python
#You can download dataset from http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236

import pandas as pd
import matplotlib.pylab as plt
from sklearn import preprocessing
from scipy.stats import skew, boxcox
import numpy as np

#First we import the data
data = pd.read_csv('data/On_Time_On_Time_Performance_2015_1.csv')

#Replace Missing Values with zero
data['AIR_TIME'].fillna(0,inplace=True)

#The next line uses scale method from scikit-learn to transform the distribution
#This will not impact Skewness Statistic calculation
#We have included this for sake of completion
airTimeOrig = preprocessing.scale(data['AIR_TIME'])

#Next We calculate Skewness using skew in spicy.stats
skness = skew(airTimeOrig)

#We draw the histograms
figure = plt.figure(figsize=(10,7))
figure.add_subplot(121)
plt.hist(airTimeOrig,facecolor='blue',alpha=0.75)
plt.xlabel("AirTime - Transformed")
plt.title("Transformed AirTime Histogram")
plt.text(2,100000,"Skewness: {0:.2f}".format(skness))

figure.add_subplot(122)
plt.boxplot(airTimeOrig)
plt.title("Large Skewness shows Right Skewed Distribution")
#plt.show()

#Note that we changed the following line to process the square roots instead of actuals
airTime = preprocessing.scale(np.sqrt(data['AIR_TIME']))
airTimeOrig = preprocessing.scale(data['AIR_TIME'])

#Next We calculate Skewness using skew in spicy.stats
skness = skew(airTime)
sknessOrig = skew(airTimeOrig)

#We draw the histograms
figure = plt.figure(figsize = (10, 7))
figure.add_subplot(131)
plt.hist(airTime,facecolor='red',alpha=0.75)
plt.xlabel("AirTime - Transformed(Using Sqrt)")
plt.title("Transformed AirTime Histogram")
plt.text(2,100000,"Skewness: {0:.2f}".format(skness))

figure.add_subplot(132)
plt.hist(airTimeOrig,facecolor='blue',alpha=0.75)
plt.xlabel("AirTime - Based on Original Flight Times")
plt.title("AirTime Histogram - Right Skewed")
plt.text(2,100000,"Skewness: {0:.2f}".format(sknessOrig))

figure.add_subplot(133)
plt.boxplot(airTime)
plt.title("Un-Skewed Distribution")
#plt.show()

#   inding the right transformation to resolve Skewness can be tedious. Box and Cox in their 1964 paper proposed
#   a statistical method to find the right transformation. They suggested using below family of transformations
#   and finding the λ:

#   $latex x^{*} = \begin{cases}\frac{x^{\lambda}-1}{\lambda} & \lambda \neq 0\\log(x) & \lambda = 0\end{cases}&s=3$

#   Notice that because of the log term, this transformation requires x values to be positive. So, if there are zero and
#   negative values, all values need to be shifted before applying this method.

#Note that we shift the values by 1 to get rid of zeros
airTimeBoxCox = preprocessing.scale(boxcox(data['AIR_TIME']+1)[0])

#Next We calculate Skewness using skew in spicy.stats
sknessBoxCox = skew(airTimeBoxCox)

#We draw the histograms
figure = plt.figure(figsize = (12, 8))
figure.add_subplot(131)
plt.hist(airTime, bins=50, facecolor='red',alpha=0.75)
plt.xlabel("AirTime - Transformed(Using Sqrt)")
plt.title("Transformed AirTime Histogram")
plt.text(2,100000,"Skewness: {0:.2f}".format(skness))

figure.add_subplot(132)
plt.hist(airTimeBoxCox, bins=50,facecolor='blue',alpha=0.75)
plt.xlabel("AirTime - Using BoxCox Transformation")
plt.title("AirTime Histogram - Un-Skewed(BoxCox)")
plt.text(2,100000,"Skewness: {0:.2f}".format(sknessBoxCox))

figure.add_subplot(133)
plt.hist(airTimeOrig, bins=50,facecolor='green',alpha=0.75)
plt.xlabel("AirTime - Based on Original Flight Times")
plt.title("AirTime Histogram - Right Skewed")
plt.text(2,100000,"Skewness: {0:.2f}".format(sknessOrig))

plt.show()