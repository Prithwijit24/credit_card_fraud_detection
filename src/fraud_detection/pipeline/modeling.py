from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from fraud_detection.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def build_training_pipeline(seed: int, max_bins: int, max_depth: int, num_trees: int) -> Pipeline:
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_index",
            handleInvalid="keep",
        )
        for column in CATEGORICAL_COLUMNS
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{column}_index" for column in CATEGORICAL_COLUMNS],
        outputCols=[f"{column}_encoded" for column in CATEGORICAL_COLUMNS],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=[f"{column}_encoded" for column in CATEGORICAL_COLUMNS] + NUMERIC_COLUMNS,
        outputCol="features",
        handleInvalid="keep",
    )
    classifier = RandomForestClassifier(
        labelCol="is_fraud",
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        seed=seed,
        maxBins=max_bins,
        maxDepth=max_depth,
        numTrees=num_trees,
        weightCol="class_weight",
    )
    return Pipeline(stages=[*indexers, encoder, assembler, classifier])


def load_model(model_path: str) -> PipelineModel:
    return PipelineModel.load(model_path)


def required_input_columns() -> list[str]:
    return [*FEATURE_COLUMNS, "is_fraud"]


def build_cv_pipeline(
    seed: int, 
    max_bins: int, 
    max_depths: list[int], 
    num_trees_list: list[int]
) -> CrossValidator:
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_index",
            handleInvalid="keep",
        )
        for column in CATEGORICAL_COLUMNS
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{column}_index" for column in CATEGORICAL_COLUMNS],
        outputCols=[f"{column}_encoded" for column in CATEGORICAL_COLUMNS],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=[f"{column}_encoded" for column in CATEGORICAL_COLUMNS] + NUMERIC_COLUMNS,
        outputCol="features",
        handleInvalid="keep",
    )
    classifier = RandomForestClassifier(
        labelCol="is_fraud",
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        seed=seed,
        maxBins=max_bins,
        weightCol="class_weight",
    )
    pipeline = Pipeline(stages=[*indexers, encoder, assembler, classifier])
    
    param_grid = (
        ParamGridBuilder()
        .addGrid(classifier.maxDepth, max_depths)
        .addGrid(classifier.numTrees, num_trees_list)
        .build()
    )
    
    evaluator = BinaryClassificationEvaluator(
        labelCol="is_fraud", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderROC"
    )
    
    return CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=seed,
    )
