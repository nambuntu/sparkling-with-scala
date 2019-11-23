import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object Kmean_Project extends App {
  def main(): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate();
    // Load the Wholesale Customers Data
    val dataset = spark.read.option("header", true).option("inferSchema", true).csv("data/kmean/Wholesale customers data.csv")
    // Select the following columns for the training set:
    // Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
    // Cal this new subset feature_data
    val feature_data = dataset.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")

    // Create a new VectorAssembler object called assembler for the feature
    // columns as the input Set the output column to be called features
    // Remember there is no Label column
    val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

    // Use the assembler object to transform the feature_data
    // Call this new data training_data
    val training_data = assembler.transform(feature_data).select("features")

    // Create a Kmeans Model with K=3
    val kmeans = new KMeans().setK(100).setSeed(1L)

    // Fit that model to the training_data
    val model = kmeans.fit(training_data)

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(training_data)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

  }

  Kmean_Project.main()
}