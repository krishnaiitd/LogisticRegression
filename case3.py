#>>>>>> Random Forest Classification on AWS reviews datasets, with Cross Validation taking equally from rating datasets, i.e dataset is based towards rating 5 <<<<<
# Import the Basic libraries
import sys
import json
import string
import random
import timeit

# Import the Spark context and config
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
# Create the SparkContext, sqlContext objects
conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Import the MLLib library need for classification
from pyspark.sql import DataFrame
from pyspark.ml.feature import HashingTF

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint


# Import the Word tokenize 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Initial the English word STOPWORDS and STEMMER
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Initialze the use variables here
numData = 500
numFeatures = 2000
maximumIteration = 100
regParamValue = 0.01
seedValue = 0

# Results sets which will be store in a file after completing all process.
results = [('Name', "Result"), ('File source name',  'LRC_1Gram_Skewness_v8.py'), ('Total number of datasets from each group',numData), ('Total number of datasets from all group', numData*5), ('Total number of Features',  numFeatures), ('Classification Parameters:', ''), ('Maximum number of iteration', maximumIteration), ('Reg Param value', regParamValue)]

# Generate unique file name, so that we do not need to change this again and again
uniqueString = '' . join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
FileName = 'LRC_1Gram_numData' + str(numData) + '_numFeatures' + str(numFeatures) + '_r' + uniqueString

# Read the json file and let's watch the first two lines
#reviewsDF = sqlContext.read.json("file:///home/cloudera/SparkML/__data/InLine.json")
#reviewsDF = sqlContext.read.json("gs://sparkrun123/reviews_Apps_for_Android.json.gz")
reviewsDF = sqlContext.read.json("gs://sparkrun123/InLine.json")

print 'Selecting only overall and review Text'
reviewsDF = reviewsDF.select("overall", "reviewText")
# Make the reviews data set persistent RDD
reviewsDF.persist()
print 'Total number of record in review datasets: ' + str(reviewsDF.count())
print 'Number of records by rating:'
print reviewsDF.groupBy('overall').count().orderBy('overall').show()

# Define the Tokenize
def tokenize(text):
        tokens = word_tokenize(text)
        lowercased = [t.lower() for t in tokens]
        no_punctuation = []
        for word in lowercased:
                punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
                no_punctuation.append(punct_removed)
        no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
        stemmed = [STEMMER.stem(w) for w in no_stopwords]
        return [w for w in stemmed if w]

print 'Clean the review text dataset'
wordsData = reviewsDF.map(lambda (label, text): (label, tokenize(text)))
# Clear reviewsDF from cache/memory
del reviewsDF
wordsData = sqlContext.createDataFrame(wordsData, ['label', 'words'])

hashingTF_1words = HashingTF(inputCol = "words", outputCol="features", numFeatures = numFeatures)

data = hashingTF_1words.transform(wordsData).select("label", "features")
del wordsData

def TrainLRCModel(trainingData, testData):
	print(type(trainingData))
	print(trainingData.take(2))

	# Map the training and testing dataset into Labeled Point
	trainingData = trainingData.map(lambda row:[LabeledPoint(row.label,row.features)])
	print('After changing the dataset type to labeled Point')
	print(type(trainingData))
	print(trainingData.take(2))

	model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=5)
	print(type(model))
	exit();
	predictions = testData.map(lambda p: (p.label, model.predict(p.features)))

	correct = predictions.filter(lambda (x, p): x == p)
	### Calculate the accuracy of the model using custom method
	accuracy = round((correct.count() / float(testData.count())) * 100, 3)
	# return the final accuracy
	return accuracy


accuracyList = [];
elapsedTimes = [];
for i in range(1,3):
	(trainData, testData) = data.randomSplit([.7, .3], seed = seedValue)
	start_time = timeit.default_timer()
	print 'Model ' + str(i)
	print 'Number of dataset in training: ' + str(trainData.count())
	print 'Number of dataset in testing: ' + str(testData.count())
	# Get the accuracy for model 1
	accuracy = TrainLRCModel(trainData, testData)
	accuracyList.append(accuracy)
	print 'Model ' + str(i) + ' accuracy ' + str(accuracy)
	elapsed = timeit.default_timer() - start_time
	print "Elapsed time: " + str(round(elapsed / 60, 2)) + ' minutes'
	elapsedTimes.append(round(elapsed / 60, 2))
	
	results.append(('Model', i))
	results.append(('Number of dataset in training', str(trainData.count())))
	results.append(('Number of dataset in testing', testData.count()))
	results.append(('Model ' + str(i) + ' accuracy', accuracy))
	results.append(("Elapsed time", str(round(elapsed / 60, 2)) + ' minutes'))

print 'All model accuracy list: '
print accuracyList
AverageAccuracy = round(sum(accuracyList) / len(accuracyList), 3)

print 'Average accuracy ' + str(AverageAccuracy) + '%'
results.append(('===Final accuracy=====',''))
results.append(('Average accuracy', str(AverageAccuracy) + '%'))
results.append(('Total time ', str(round(sum(elapsedTimes), 2)) + ' minutes'))

results = sc.parallelize(results)
# Store the actual and predicted classes in a file with name : FileName
def toCSVLine(data):
  return ','. join(str(d) for d in data)

lines = results.map(toCSVLine)
lines.saveAsTextFile("gs://spark_123/results/" + FileName)
#lines.saveAsTextFile("file:///home/cloudera/SparkML/__data/" + FileName)
sc.stop()
