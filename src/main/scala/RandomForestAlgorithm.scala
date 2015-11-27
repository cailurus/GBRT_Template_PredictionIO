package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

//import org.apache.spark.mllib.classification.NaiveBayes
//import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

/*
case class AlgorithmParams(
  lambda: Double
) extends Params
*/
case class RandomForestAlgorithmParams(
    numClasses: Int,
    numTrees: Int,
    featureSubsetStrategy: String,
    impurity: String,
    maxDepth: Int,
    maxBins: Int
) extends Params

class RandomForestAlgorithm(val ap: RandomForestAlgorithmParams)
    extends P2LAlgorithm[PreparedData, RandomForestModel, Query, PredictedResult] {

    def train(sc: SparkContext, data: PreparedData): RandomForestModel = {
        val categoricalFeaturesInfo = Map[Int, Int]()
        RandomForest.trainClassifier(
            data.labeledPoints,
            ap.numClasses,
            categoricalFeaturesInfo,
            ap.numTrees,
            ap.featureSubsetStrategy,
            ap.impurity,
            ap.maxDepth,
            ap.maxBins)
    }

    def predict(model: RandomForestModel, query: Query): PredictedResult = {
        val label = model.predict(Vectors.dense(
            query.attr0, query.attr1, query.attr2
            ))
        new PredictedResult(label)
    }
}

/*

// extends P2LAlgorithm because the MLlib's NaiveBayesModel doesn't contain RDD.
class NaiveBayesAlgorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, NaiveBayesModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): NaiveBayesModel = {
    // MLLib NaiveBayes cannot handle empty training data.
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")

    NaiveBayes.train(data.labeledPoints, ap.lambda)
  }

  def predict(model: NaiveBayesModel, query: Query): PredictedResult = {
    val label = model.predict(Vectors.dense(
      Array(query.attr0, query.attr1, query.attr2)
    ))
    new PredictedResult(label)
  }

}
*/
