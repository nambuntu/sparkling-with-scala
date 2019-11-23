package work.inlumina.sparkwscala.regression

import org.apache.spark.ml.regression.LinearRegression
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

    val lr = new LinearRegression()
    val lrModel = lr.fit(output)

    val trainingSummary = lrModel.summary

    trainingSummary.residuals.show()
    println("Features for prediction: ")
    for (ind <- Range(1, colnames.length)) {
      print(colnames(ind) + ",")
    }
    println()
    println("coefficient of determination R2: " + trainingSummary.r2)
    trainingSummary.predictions.select($"features", $"label", $"prediction").show()


  }

  HousePricing.main()
}
