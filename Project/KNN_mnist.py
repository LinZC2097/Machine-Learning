from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import sklearn
import csv
import time


if __name__ == '__main__':
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # train_data_int = []
    # train_label_int = []
    # test_data_int = []
    with open("/Users/marsscho/Desktop/data_mnist.csv", 'r', encoding="UTF-8") as csvTraData:
        readTraData = csv.reader(csvTraData)
        next(readTraData)
        for row in readTraData:
            train_data.append(list(map(int, row[1:])))
            train_label.append(int(row[0]))

    print(len(train_data[0]))
    print(len(train_data))

    with open("/Users/marsscho/Desktop/test_mnist.csv", 'r', encoding="UTF-8") as csvTestData:
        readTestData = csv.reader(csvTestData)
        next(readTestData)

        for row in readTestData:
            test_data.append(list(map(int, row)))

    print(len(test_data[0]))
    print(len(test_data))

    # (trainData, testData, trainLabel, testLabel) = train_test_split(train_data, train_label, test_size=0.2, random_state=84)


    print("the length of the train data is %d" % len(train_data))
    print("the length of the test data is %d" % len(test_data))


    with open("/Users/marsscho/Desktop/test_label.csv", mode='a', encoding="UTF-8") as csvTestLabel:
        writerTestLabel = csv.writer(csvTestLabel)
        writerTestLabel.writerow(['ImageID', 'Label'])

    # for k in range(1, 100, 2):
    #     start_time = time.time()
    #
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", k)
    #     # print("the start time is %.2f" % start_time)
    #     neigh = KNeighborsClassifier(n_neighbors=k)
    #     neigh.fit(trainData, trainLabel)
    #     test_label = []
    #     test_label.append(k)
    #
    #     score = neigh.score(testData, testLabel)
    #     print(score)
    #     test_label.append(score)
    #     # for val in testData:
    #     #     prediction = neigh.predict([val])
    #     #     test_label.append(prediction[0])
    #     #     print(prediction[0])
    #     test_label.append(time.time() - start_time)
    #     print("the RUN time is %.2f" % (test_label[2]))
    #
    #     with open("/Users/marsscho/Desktop/test_label_accuracy_unscaled.csv", mode='a', encoding="UTF-8") as csvTestLabel:
    #         writerTestLabel = csv.writer(csvTestLabel)
    #         writerTestLabel.writerow(test_label)

    print("KNN start")
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data, train_label)

    for i in range(len(test_data)):
        prediction = neigh.predict([test_data[i]])
        print(i, '\t\t', prediction)

        with open("/Users/marsscho/Desktop/test_label.csv", mode='a', encoding="UTF-8") as csvTestLabel:
            writerTestLabel = csv.writer(csvTestLabel)
            writerTestLabel.writerow([i, prediction[0]])
