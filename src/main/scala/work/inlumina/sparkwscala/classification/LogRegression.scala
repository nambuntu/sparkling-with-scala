package work.inlumina.sparkwscala.classification

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.hour

//////////////////////////////////////////////
// LOGISTIC REGRESSION PROJECT //////////////
////////////////////////////////////////////

//  In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
//  This data set contains the following features:
//    'Daily Time Spent on Site': consumer time on site in minutes
//    'Age': cutomer age in years
//    'Area Income': Avg. Income of geographical area of consumer
//    'Daily Internet Usage': Avg. minutes a day consumer is on the internet
//    'Ad Topic Line': Headline of the advertisement
//    'City': City of consumer
//    'Male': Whether or not consumer was male
//    'Country': Country of consumer
//    'Timestamp': Time at which consumer clicked on Ad or closed window
//    'Clicked on Ad': 0 or 1 indicated clicking on Ad
object LogRegression extends App {
  def main(): Unit = {
    ////////////////////////
    /// GET THE DATA //////
    //////////////////////
    // Import SparkSession and Logistic Regression

    // Optional: Use the following code below to set the Error reporting

    // Create a Spark Session
    val spark = SparkSession.builder().master("local[*]").appName("LogisticRegression").getOrCreate()
    import spark.implicits._
    // Use Spark to read in the Advertising csv file.
    val data = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("data/classification/advertising.csv")

    // Print the Schema of the DataFrame
    data.printSchema()

    ///////////////////////
    /// Display Data /////
    /////////////////////

    // Print out a sample row of the data (multiple ways to do this)
    for (header <- data.columns) {
      print(header + ", ")
    }
    println()
    for (row <- data.head(5)) {
      println(row)
    }

    ////////////////////////////////////////////////////
    //// Setting Up DataFrame for Machine Learning ////
    //////////////////////////////////////////////////

    //   Do the Following:
    //    - Rename the Clicked on Ad column to "label"
    //    - Grab the following columns "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
    //    - Create a new column called Hour from the Timestamp containing the Hour of the click
    val timedata = data.withColumn("Hour", hour(data("Timestamp")))
    var logregdata = timedata.select(data("Clicked on Ad").as("label"),
      $"Daily Time Spent on Site",
      $"Age",
      $"Area Income",
      $"Daily Internet Usage",
      $"Timestamp",
      $"Male",
      $"Hour")
    logregdata = logregdata.na.drop()
    // Import VectorAssembler and Vectors

    // Create a new VectorAssembler object called assembler for the feature
    // columns as the input Set the output column to be called features
    val assembler = new VectorAssembler()
      .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Hour"))
      .setOutputCol("features")
    // Use randomSplit to create a train test split of 70/30
    val Array(training, test) = logregdata.randomSplit(Array(0.8, 0.2), seed = 12345)

    ///////////////////////////////
    // Set Up the Pipeline ///////
    /////////////////////////////

    // Import Pipeline
    // Create a new LogisticRegression object called lr
    val lr = new LogisticRegression()

    // Create a new pipeline with the stages: assembler, lr
    val pipeline = new Pipeline().setStages(Array(assembler, lr))
    // Fit the pipeline to training set.
    val model = pipeline.fit(training)
    // Get Results on Test Set with transform
    val results = model.transform(test)

    ////////////////////////////////////
    //// MODEL EVALUATION /////////////
    //////////////////////////////////

    // For Metrics and Evaluation import MulticlassMetrics

    // Convert the test results to an RDD using .as and .rdd

    // Instantiate a new MulticlassMetrics object

    // Print out the Confusion matrix
    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

    // Instantiate a new MulticlassMetrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Print out the Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)
    println("Accuracy: " + metrics.accuracy)

    spark.stop()
  }

  LogRegression.main()
}
