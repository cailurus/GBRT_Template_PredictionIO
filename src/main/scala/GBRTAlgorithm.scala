package org.template.classification

import org.apache.predictionio.controller.P2LAlgorithm
import org.apache.predictionio.controller.Params

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class GBRTAlgorithmParams(
    numIterations: Int,
    numClasses: Int,
    maxDepth: Int
) extends Params

class GBRTAlgorithm(val ap: GBRTAlgorithmParams)
    extends P2LAlgorithm[PreparedData, GradientBoostedTreesModel, Query, PredictedResult]{

    def train(sc: SparkContext, data: PreparedData): GradientBoostedTreesModel = {
        val boostingStrategy = BoostingStrategy.defaultParams("Classification")
        boostingStrategy.numIterations = ap.numIterations
        boostingStrategy.treeStrategy.numClasses = ap.numClasses
        boostingStrategy.treeStrategy.maxDepth = ap.maxDepth
        boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

        GradientBoostedTrees.train(data.labeledPoints, boostingStrategy)
    }

    def predict(model: GradientBoostedTreesModel, query: Query): PredictedResult = {
        val label = model.predict(Vectors.dense(
            query.attr0, query.attr1, query.attr2
            ))
        new PredictedResult(label)
    }
}
