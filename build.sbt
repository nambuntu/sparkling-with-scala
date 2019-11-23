lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "work.inlumina",
      scalaVersion := "2.12.8",
      version := "0.1.0-SNAPSHOT"
    )),
    name := "sparkling-with-scala",
    libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.4",
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.4",
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.4"
  )
