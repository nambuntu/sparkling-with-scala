package work.inlumina.sparkwscala.df

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

// http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.functions

object DateTimestamp extends App {
  val spark = SparkSession.builder().master("local[*]").getOrCreate() // Start a simple Spark Session

  import spark.implicits._

  val df = spark.read.option("header", "true").option("inferSchema", "true").csv("data/df/CitiGroup2006_2008")

  df.select(year(df("Date"))).show()

  val df2 = df.withColumn("Year", year(df("Date")))
  val dfavg = df2.groupBy("Year").mean()

  dfavg.select($"Year", $"avg(Close)").show()

  val dfmin = df2.groupBy("Year").min()

  dfmin.select($"Year", $"min(Close)").show()
}
