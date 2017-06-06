# -*- coding: utf-8 -*-
"""
Trying out the scikit KNN API

Created on Tue Jun  6 15:07:59 2017

@author: Praveen S
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from numpy import mean

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

k = 7

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, 
                                        test_size = 0.33, random_state=42, )

#initialize KNN clasifier object
knn = KNeighborsClassifier(n_neighbors=k)

#train the 
knn.fit(X_train, y_train)
result = knn.predict(X_test)

output = zip(iris.target_names[result],
      iris.target_names[y_test])

[print("Predicted '{0}' \t > Actual '{1}' \t {2}".format(predicted, actual,
       predicted == actual)) 
    for (predicted, actual) in output]

print(mean(result == y_test))
