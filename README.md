# Multi-class Logistic Regression implementation on PySpark and issues discussion

## Case 1:

I simple used the Logistic regression pipeline.

### Library used

``` 
from pyspark.ml.feature import HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
```

### Dataset type and calling of LogisticRegression pipeline
```
print(type(trainingData))
print(trainingData.take(2))
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=maximumIteration, 	regParam=regParamValue)
pipeline = Pipeline(stages=[lr])
# Train model	
model = pipeline.fit(trainingData)
```

### Error output with dataset type from print statements

```
<class 'pyspark.sql.dataframe.DataFrame'>
[Row(label=2.0, features=SparseVector(2000, {51: 1.0, 160: 1.0, 341: 1.0, 417: 1.0, 561: 1.0, 656: 1.0, 863: 1.0, 939: 1.0, 1021: 1.0, 13
24: 1.0, 1433: 1.0, 1573: 1.0, 1604: 1.0, 1720: 1.0})), Row(label=2.0, features=SparseVector(2000, {103: 1.0, 801: 1.0, 1021: 1.0, 1170: 
1.0, 1222: 1.0, 1508: 1.0, 1778: 1.0, 1858: 2.0}))]
16/08/27 07:31:12 ERROR org.apache.spark.ml.classification.LogisticRegression: Currently, LogisticRegression with ElasticNet in ML packag
e only supports binary classification. Found 6 in the input dataset.
Traceback (most recent call last):
  File "/home/LR/LogisticRegression/case1.py", line 110, in <module>
    accuracy = TrainLRCModel(trainData, testData)
  File "/home/LR/LogisticRegression/case1.py", line 90, in TrainLRCModel
    model = pipeline.fit(trainingData)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/ml/pipeline.py", line 69, in fit
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/ml/pipeline.py", line 213, in _fit
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/ml/pipeline.py", line 69, in fit
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/ml/wrapper.py", line 133, in _fit
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/ml/wrapper.py", line 130, in _fit_java
  File "/usr/lib/spark/python/lib/py4j-0.9-src.zip/py4j/java_gateway.py", line 813, in __call__
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/sql/utils.py", line 45, in deco
  File "/usr/lib/spark/python/lib/py4j-0.9-src.zip/py4j/protocol.py", line 308, in get_return_value
py4j.protocol.Py4JJavaError: An error occurred while calling o100.fit.
: org.apache.spark.SparkException: Currently, LogisticRegression with ElasticNet in ML package only supports binary classification. Found
 6 in the input dataset.
        at org.apache.spark.ml.classification.LogisticRegression.train(LogisticRegression.scala:290)
        at org.apache.spark.ml.classification.LogisticRegression.train(LogisticRegression.scala:159)
        at org.apache.spark.ml.Predictor.fit(Predictor.scala:90)
        at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.lang.reflect.Method.invoke(Method.java:498)
        at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:231)
        at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:381)
        at py4j.Gateway.invoke(Gateway.java:259)
        at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:133)
        at py4j.commands.CallCommand.execute(CallCommand.java:79)
        at py4j.GatewayConnection.run(GatewayConnection.java:209)
at java.lang.Thread.run(Thread.java:745)
```



## Case 2:

I simply use `LogisticRegressionWithLBFGS` as follows:

### Library used
```
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD
```

### Calling of LogisticRegressionWithLBFGS, I am also printing the dataset type

```
print(type(trainingData))
print(trainingData.take(2))
model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=5)
print(type(model))
```


### Error output with dataset type

```
<class 'pyspark.sql.dataframe.DataFrame'>
[Row(label=2.0, features=SparseVector(2000, {51: 1.0, 160: 1.0, 341: 1.0, 417: 1.0, 561: 1.0, 656: 1.0, 863: 1.0, 939: 1.0, 1021: 1.0, 13
24: 1.0, 1433: 1.0, 1573: 1.0, 1604: 1.0, 1720: 1.0})), Row(label=2.0, features=SparseVector(2000, {103: 1.0, 801: 1.0, 1021: 1.0, 1170: 
1.0, 1222: 1.0, 1508: 1.0, 1778: 1.0, 1858: 2.0}))]
Traceback (most recent call last):
  File "/home/LR/LogisticRegression/case2.py", line 109, in <module>
    accuracy = TrainLRCModel(trainData, testData)
  File "/home/LR/LogisticRegression/case2.py", line 88, in TrainLRCModel
    model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=5)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/mllib/classification.py", line 382, in train
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/mllib/regression.py", line 206, in _regression_train_wrapper
TypeError: data should be an RDD of LabeledPoint, but got <class 'pyspark.sql.types.Row'>
root@sparkrun-m:/home/LR/LogisticRegression#
```

## case 3:
Since the case 2 giving error of input data type, going to change the training data type from  Data Frame to LabeledPoint

### Library used
```
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
```
	print(type(trainingData))
	print(trainingData.take(2))

	# Map the training and testing dataset into Labeled Point
	trainingData = trainingData.map(lambda row:[LabeledPoint(row.label,row.features)])
	print('After changing the dataset type to labeled Point')
	print(type(trainingData))
	print(trainingData.take(2))

	model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=5)
	print(type(model))
```

### Error output with dataset type
```
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
```







