package work.inlumina.sparklingwithscala.df

import org.apache.spark.sql.SparkSession

object MissingData extends App {
  val spark = SparkSession.builder().master("local[*]").getOrCreate() // Start a simple Spark Session

  val df = spark.read.option("header", "true").option("inferSchema", "true").csv("data/df/ContainsNull.csv")

  df.printSchema()
  df.show()

  df.na.drop(2).show()

  val df2 = df.na.fill("New Name", Array("Name"))
  df2.na.fill(100, Array("Sales")).show()
}
