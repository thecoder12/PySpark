from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Customers').getOrCreate()
from pyspark.ml.regression import LinearRegression

dataset = spark.read.csv('Ecommerce_Customers.csv', header=True, inferSchema=True)
dataset.show(3)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

featureassembler = VectorAssembler(inputCols=['Avg Session Length','Time on App','Time on Website','Length of Membership'],outputCol='Independent feature')
output = featureassembler.transform(dataset)
output.show(3)

finalized_data = output.select('Independent feature','Yearly Amount Spent')
finalized_data.show(3)


train_data, test_data = finalized_data.randomSplit([0.75, 0.25])

regressor = LinearRegression(featuresCol='Independent feature', labelCol='Yearly Amount Spent')
regressor = regressor.fit(train_data)
print(regressor.coefficients, regressor.intercept)

pred_result = regressor.evaluate(test_data)
pred_result.predictions.show()



