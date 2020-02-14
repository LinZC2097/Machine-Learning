from pyspark import SQLContext
from pyspark.sql import SparkSession
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint


def main():
    spark = SparkSession.builder.appName("Word Count").getOrCreate()
    sc = spark.sparkContext

    sqlContext = SQLContext(sc)

    spark = SparkSession.builder.appName('ml-bank').getOrCreate()
    data = spark.read.csv('/Users/marsscho/Desktop/6364 Machine Learning/assignment/hw2/Pokemon.csv'
                          , header=True, inferSchema=True)
    data.printSchema()

    data = sqlContext.read.format('com.databricks.spark.csv')\
        .options(header='true', inferschema='true')\
        .load('/Users/marsscho/Desktop/6364 Machine Learning/assignment/hw2/Pokemon.csv')
    data.show()

    result = []

    rdd = data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

    (trainingData, testData) = rdd.randomSplit([0.7, 0.3])

    model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={0: 19, 1: 20},
                                         impurity='entropy', maxDepth=3, maxBins=32)

    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count()

    # lambda lp: lp[0] != lp[1]).count() / float(testData.count())
    print("test err", testErr)
    print("test data count:", testData.count())
    print("test data count:", testData.count())
    print('Test precision = ' + str(1 - testErr / float(testData.count())))
    # result.append((depth, testErr))
    print('Learned classification tree model:')
    print(model.toDebugString())


    # for depth in range(1, 20):
    #
    #     model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={0: 19, 1: 20},
    #                                          impurity='entropy', maxDepth=depth, maxBins=32)
    #
    #     predictions = model.predict(testData.map(lambda x: x.features))
    #     labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    #     testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count()
    #
    #     # lambda lp: lp[0] != lp[1]).count() / float(testData.count())
    #     print("test err", testErr)
    #     print("test data count:", testData.count())
    #     print("test data count:", testData.count())
    #     print('Test precision = ' + str(1 - testErr/float(testData.count())))
    #     result.append((depth, testErr))
    #     print('Learned classification tree model:')
    #     print(model.toDebugString())
    for val in result:
        print(val)



if __name__ == '__main__':
    main()
