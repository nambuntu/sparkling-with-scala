package work.inlumina.sparklingwithscala.recommendation

import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Recommender_System extends App {
  val DATE_TIME_FORMAT = "yyyy-MM-dd HH:mm:ss"

  def main(): Unit = {
    var now = Calendar.getInstance().getTime()
    val format = new SimpleDateFormat(DATE_TIME_FORMAT)
    println("Start, Datetime now: " + format.format(now))

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate();
    import spark.implicits._

    val ratings = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .format("csv")
      .load("W:/learning/ml-20m/ratings.csv")

    ratings.printSchema()

    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    println("training count: " + training.count())
    println("test count: " + test.count())

    val als = new ALS().setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val model = als.fit(training)
    val predictions = model.transform(test)

    val error = predictions.select(abs($"rating" - $"prediction"))

    predictions.show()

    predictions.where(abs($"prediction" - $"rating") < 0.1).show(200)
    predictions.write.csv("W:/learning/ml-20m/prediction_result.csv")
    //error.na.drop().describe().show()
    now = Calendar.getInstance().getTime()
    println("Finished, Datetime now: " + format.format(now))
  }

  Recommender_System.main()
}
