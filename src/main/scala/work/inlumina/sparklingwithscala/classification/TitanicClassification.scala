package work.inlumina.sparklingwithscala.classification

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object TitanicClassification extends App {
  def main(): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("Titanic Classification").getOrCreate()
    import spark.implicits._

    val data = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("data/classification/titanic.csv")

    data.printSchema()

    println("Sample data: ")
    for (row <- data.head(5)) {
      println(row)
    }

    val logregdataall = (data.select(data("Survived").as("label"),
      $"Pclass",
      $"Name",
      $"Sex",
      $"Age",
      $"SibSp",
      $"Parch",
      $"Fare",
      $"Embarked"))

    val logregdata = logregdataall.na.drop()

    //Convert String into Numerical values
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

    //Convert Numerical values into One hot encoding

    val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
    val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVec")

    //Label,feature
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkedVec"))
      .setOutputCol("features")

    val Array(training, test) = logregdata.randomSplit(Array(0.75, 0.25), seed = 12345)

    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(
      Array(genderIndexer, embarkedIndexer, genderEncoder, embarkedEncoder, assembler, lr))
    val model = pipeline.fit(training)
    val results = model.transform(test)

    results.show()

    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion matrics")
    println(metrics.confusionMatrix)
    println("Accuracy: " + metrics.accuracy)
    spark.stop()
  }

  TitanicClassification.main()
}
