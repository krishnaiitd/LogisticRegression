root@sparkrun-m:/home/LR/LogisticRegression# spark-submit case3.py 
Selecting only overall and review Text                                          
Total number of record in review datasets: 10                                   
Number of records by rating:
+-------+-----+                                                                 
|overall|count|
+-------+-----+
|    1.0|    2|
|    2.0|    2|
|    3.0|    1|
|    4.0|    3|
|    5.0|    2|
+-------+-----+
None
Clean the review text dataset
Model 1
Number of dataset in training: 6                                                
Number of dataset in testing: 4
<class 'pyspark.sql.dataframe.DataFrame'>
[Row(label=2.0, features=SparseVector(2000, {51: 1.0, 160: 1.0, 341: 1.0, 417: 1.0, 561: 1.0, 656: 1.0, 863: 1.0, 939: 1.0, 1021: 1.0, 1
324: 1.0, 1433: 1.0, 1573: 1.0, 1604: 1.0, 1720: 1.0})), Row(label=2.0, features=SparseVector(2000, {103: 1.0, 801: 1.0, 1021: 1.0, 1170
: 1.0, 1222: 1.0, 1508: 1.0, 1778: 1.0, 1858: 2.0}))]
After changing the dataset type to labeled Point
<class 'pyspark.rdd.PipelinedRDD'>
[[LabeledPoint(2.0, (2000,[51,160,341,417,561,656,863,939,1021,1324,1433,1573,1604,1720],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.
0,1.0,1.0]))], [LabeledPoint(2.0, (2000,[103,801,1021,1170,1222,1508,1778,1858],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0]))]]
Traceback (most recent call last):
  File "/home/LR/LogisticRegression/case3.py", line 115, in <module>
    accuracy = TrainLRCModel(trainData, testData)
  File "/home/LR/LogisticRegression/case3.py", line 94, in TrainLRCModel
    model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=5)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/mllib/classification.py", line 381, in train
AttributeError: 'list' object has no attribute 'features'
root@sparkrun-m:/home/LR/LogisticRegression#
