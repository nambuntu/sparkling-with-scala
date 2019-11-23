package work.inlumina.sparklingwithscala.kmean

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession

object Kmean_Example extends App {
  def main(): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate();
    val dataset = spark.read.format("libsvm").load("data/kmean/sample_kmeans_data.txt")
    val kmeans = new KMeans().setK(2).setSeed(1L)

    val model = kmeans.fit(dataset);
    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Error = $WSSSE")

    println("Cluster Center: ")
    model.clusterCenters.foreach(println)
  }

  Kmean_Example.main();
}

