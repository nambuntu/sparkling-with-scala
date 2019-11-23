package work.inlumina.sparkwscala.evaluation

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.{Row, SparkSession}

object SimpleLoad01 extends App {
  def main(): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    // Prepare test documents, which are unlabeled (id, text) tuples.
    val test = spark.createDataFrame(Seq(
      (14L, "spark i j k"),
      (15L, "l m n"),
      (16L, "mapreduce spark"),
      (17L, "apache hadoop"),
      (18L, "hgas spar k kas so"))).toDF("id", "text")

    val path = "data/evaluation/best-model-example-01"
    println(s"Loading a trained model from: $path")
    val cvModel = CrossValidatorModel.load(path)

    println("Result: ")
    // Make predictions on test documents. cvModel uses the best model found (lrModel).
    cvModel.transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach {
        case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
          println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }

    spark.stop()
    println("Done.")
  }

  SimpleLoad01.main()
}
