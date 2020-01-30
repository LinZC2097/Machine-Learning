from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import sklearn
import csv


if __name__ == '__main__':
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # train_data_int = []
    # train_label_int = []
    # test_data_int = []
    with open("/Users/marsscho/Desktop/training_data_40000.csv", 'r', encoding="UTF-8") as csvTraData:
        readTraData = csv.reader(csvTraData)
        next(readTraData)
        for row in readTraData:
            train_data.append(list(map(int, row[1:])))
            train_label.append(int(row[0]))

    print(len(train_data[0]))
    print(len(train_data))

    with open("/Users/marsscho/Desktop/testing_data.csv", 'r', encoding="UTF-8") as csvTestData:
        readTestData = csv.reader(csvTestData)
        next(readTestData)
        for row in readTestData:
            test_data.append(list(map(int, row)))

    print(len(test_data[0]))
    print(len(test_data))


    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_data, train_label)



    with open("/Users/marsscho/Desktop/test_label.csv", mode='w', encoding="UTF-8") as csvTestLabel:
        writerTestLabel = csv.writer(csvTestLabel)

        for val in test_data:
            prediction = neigh.predict([val])
            writerTestLabel.writerow(prediction)
            print(prediction)








