package work.inlumina.sparklingwithscala.evaluation

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

/**
 * Predicting house pricing base on some linear regression features.
 */
object HousePricing extends App {
  def main(): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate() // Start a simple Spark Session
    import spark.implicits._
    val path = "data/regression/Clean-USA-Housing.csv"
    val data = spark.read.option("header", true).option("inferSchema", true).format("csv").load(path)

    // Check out the Data
    data.printSchema()

    // See an example of what the data looks like
    // by printing out a Row
    val colnames = data.columns
    val firstrow = data.head(1)(0)
    println("\n")
    println("Example Data Row")
    for (ind <- Range(1, colnames.length)) {
      println(colnames(ind))
      println(firstrow(ind))
      println("\n")
    }

    ////////////////////////////////////////////////////
    //// Setting Up DataFrame for Machine Learning ////
    //////////////////////////////////////////////////

    // A few things we need to do before Spark can accept the data!
    // It needs to be in the form of two columns
    // ("label","features")

    // This will allow us to join multiple feature columns
    // into a single column of an array of feautre values
    import org.apache.spark.ml.feature.VectorAssembler

    // Rename Price to label column for naming convention.
    // Grab only numerical columns from the data
    val df = data.select(data("Price").as("label"), $"Avg Area Income", $"Avg Area House Age", $"Avg Area Number of Rooms", $"Area Population")

    // An assembler converts the input values to a vector
    // A vector is what the ML algorithm reads to train a model
    // Set the input columns from which we are supposed to read the values
    // Set the name of the column where the vector will be stored
    val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income", "Avg Area House Age", "Avg Area Number of Rooms", "Area Population")).setOutputCol("features")

    // Use the assembler to transform our DataFrame to the two columns
    val output = assembler.transform(df).select($"label", $"features")
    output.show() //Check 

    // Create an array of the training and test data
    val Array(training, test) = output.select("label", "features").randomSplit(Array(0.9, 0.1), seed = 12345)

    val lr = new LinearRegression()
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(1000, 0.001))
      .build()

    // In this case the estimator is simply the linear regression.
    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // 80% of the data will be used for training and the remaining 20% for validation.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    // You can then treat this object as the new model and use fit on it.
    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)

    //////////////////////////////////////
    // EVALUATION USING THE TEST DATA ///
    ////////////////////////////////////

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    model.transform(test).select("features", "label", "prediction").show(1000)

    // Check out the metrics
    for (param <- model.validationMetrics) {
      println(param)
    }

    spark.stop()
  }

  HousePricing.main()
}
