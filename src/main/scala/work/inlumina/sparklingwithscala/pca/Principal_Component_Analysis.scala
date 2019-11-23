package work.inlumina.sparklingwithscala.pca

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object Principal_Component_Analysis {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate();
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val pca = (new PCA().setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(df))

    val pcaDF = pca.transform(df)
    val result = pcaDF.select("pcaFeatures")
    result.show()
  }

  Principal_Component_Analysis.main(null)
}
