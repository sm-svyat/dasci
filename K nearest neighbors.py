import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from tools import printf #print with separation

'''
All files need to be converted from R.data format first. The can be done by opening up R console and issuing the following commands:
load("/home/sam/Documents/BDS/APM/AppliedPredictiveModeling/data/segmentationOriginal.RData") ('Route dir of .RData File)
write.csv(segmentationOriginal, file = "/home/sam/Documents/BDS/APM/segmentation_original.csv")
'''

if __name__ == '__main__':

    twoClassData = pd.read_csv('data/twoClassData.csv')
    twoClassData.columns = ['ID', 'PredictorA', 'PredictorB', 'Classes']

    printf('Columns of twoClassData\n\n', twoClassData.columns)
    printf('First five rows of twoClassData\n\n', twoClassData.head())

    predictors = twoClassData[['PredictorA', 'PredictorB']]
    classes = twoClassData.Classes

    # Split arrays or matrices into random train and test subsets. Test size 20%
    predictors_train, predictors_test, classes_train, classes_test = train_test_split(predictors, classes,
                                                                                      test_size=0.2, random_state=42)
    plt.figure(figsize=(10, 6))
    plt.plot(twoClassData.PredictorA[twoClassData.Classes == 'Class1'], twoClassData.PredictorB[twoClassData.Classes == 'Class1'],
             '^r', label='$Class$ $1$', alpha=0.6)
    plt.plot(twoClassData.PredictorA[twoClassData.Classes == 'Class2'], twoClassData.PredictorB[twoClassData.Classes == 'Class2'],
             'sb', label='$Class$ $2$', alpha=0.6)
    plt.xlabel('Predictor A')
    plt.ylabel('Predictor B')
    plt.grid()
    plt.legend()
    plt.show()

    knn=neighbors.KNeighborsClassifier()

    # we create an instance of Neighbours Classifier and fit the data.
    knn.fit(predictors_train, classes_train)

    #Probabilities for Class1 and Class 2 are given by proba function
    printf('Probabilities for Class1 and Class 2\n\n', knn.predict_proba(predictors_test)[:5])


    #Whereas class assignments are given by raw predict function
    comparison = pd.DataFrame({'PredictorA' : predictors_test['PredictorA'],
                         'PredictorB' : predictors_test['PredictorB'],
                         'PredictClass' : knn.predict(predictors_test),
                         'Class' : classes_test})
    printf('Comparison of predicted and test values\n\n', comparison)

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(comparison.PredictorA[comparison.Class == 'Class1'], comparison.PredictorB[comparison.Class == 'Class1'],
             '^r', label='$Class$ $1$', alpha=0.6)
    plt.plot(comparison.PredictorA[comparison.Class == 'Class2'], comparison.PredictorB[comparison.Class == 'Class2'],
             'sb', label='$Class$ $2$', alpha=0.6)
    plt.title('Default distribution')
    plt.xlabel('Predictor A')
    plt.ylabel('Predictor B')
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(comparison.PredictorA[comparison.PredictClass == 'Class1'],
             comparison.PredictorB[comparison.PredictClass == 'Class1'], '^r', label='$Class$ $1$', alpha=0.6)
    plt.plot(comparison.PredictorA[comparison.PredictClass == 'Class2'],
             comparison.PredictorB[comparison.PredictClass == 'Class2'], 'sb', label='$Class$ $2$', alpha=0.6)
    plt.title('Predict distribution')
    plt.xlabel('Predictor A')
    plt.ylabel('Predictor B')
    plt.grid()
    plt.legend()

    plt.show()

    #To implement cross validation in python, you can call the predictor directly with the X-val in cross_val_score
    #Here we are running knn on the above data, with a 10 fold x-val
    scores = cross_val_score(knn, predictors, classes, cv=3)

    #The function returns an accuracy score for each run of the classifier.
    printf('Accuracy score for each run of the classifier\n\n', scores)