from pyspark import SQLContext, SparkContext, conf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
import pandas as pd


def main():
    spark = SparkSession.builder.appName("Word Count").getOrCreate()
    sc = spark.sparkContext

    sqlContext = SQLContext(sc)

    spark = SparkSession.builder.appName('ml-bank').getOrCreate()
    data = spark.read.csv('/Users/marsscho/Desktop/6364 Machine Learning/assignment/hw2/Pokemon.csv', header=True, inferSchema=True)
    data.printSchema()

    data = sqlContext.read.format('com.databricks.spark.csv')\
        .options(header='true', inferschema='true')\
        .load('/Users/marsscho/Desktop/6364 Machine Learning/assignment/hw2/Pokemon.csv')
    data.show()

    # indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").setHandleInvalid("keep").fit(data)
    #             for column in ['Type1', 'Type2']]
    #
    #
    # pipeline = Pipeline(stages=indexers)
    # df_r = pipeline.fit(data).transform(data)
    # print(type(df_r))
    # # df_r.drop(['Type1', 'Type2'])
    # df_r.drop(df_r.Type1).collect()
    # df_r.show()
    # print(df_r.Type1)

    rdd = data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

    (trainingData, testData) = rdd.randomSplit([0.7, 0.3])

    model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={0, 1},
                                         impurity='gini', maxDepth=5, maxBins=32)

    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification tree model:')
    print(model.toDebugString())

    # Save and load model
    # model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
    # sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")



if __name__ == '__main__':
    main()
