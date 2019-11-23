package work.inlumina.sparkwscala.classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object ClassificationDocumentation extends App {
  def main(): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("LogisticRegressionWithElasticNetExample")
      .getOrCreate()

    // $example on$
    // Load training data
    val training = spark.read.format("libsvm").load("data/classification/sample_libsvm_data.txt")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    // $example off$

    spark.stop()
  }

  ClassificationDocumentation.main()
}
