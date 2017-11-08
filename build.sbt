import AssemblyKeys._

assemblySettings

name := "template-scala-parallel-classification"

organization := "org.apache.predictionio"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.predictionio"    %% "apache-predictionio-core"          % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core"    % "1.3.1" % "provided",
  "org.apache.spark" %% "spark-mllib"   % "1.3.1" % "provided")
