package work.inlumina.sparkwscala.regression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

/**
 * Predicting ecommerce spending based on some linear regression features.
 */
object EcommSpending extends App {
  def main(): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("My spark app").getOrCreate()
    import spark.implicits._

    val data = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("data/regression/Clean-Ecommerce.csv")
    val colnames = data.columns
    data.printSchema()
    for (row <- data.head(5)) {
      println(row)
    }

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "Avg Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership"))
      .setOutputCol("features")

    val df = data.select(data("Yearly Amount Spent").as("label"),
      $"Email",
      $"Avatar",
      $"Avg Session Length",
      $"Time on App",
      $"Time on Website",
      $"Length of Membership")

    val output = assembler.transform(df).select($"features", $"label")
    output.show()

    val lr = new LinearRegression()
    val lrModel = lr.fit(output)

    val trainingSummary = lrModel.summary

    trainingSummary.residuals.show()
    println("Features for prediction: ")
    for (ind <- Range(1, colnames.length)) {
      print(colnames(ind) + ",")
    }
    println()
    println("R2: " + trainingSummary.r2)
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    trainingSummary.predictions.select($"features", $"label", $"prediction").show()
  }

  EcommSpending.main()
}
