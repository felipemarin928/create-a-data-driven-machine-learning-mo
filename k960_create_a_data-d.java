package com.k960.ml.controller;

import java.util.Map;
import java.util.Properties;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.k960.ml.config.ModelConfig;
import com.k960.ml.exceptions.ModelException;
import com.k960.ml.utils.FileUtils;
import com.k960.ml.utils.SparkUtils;

public class DataDrivenMachineLearningController {

    private SparkSession sparkSession;
    private ModelConfig modelConfig;
    private Properties properties;

    public DataDrivenMachineLearningController(Properties properties) {
        this.properties = properties;
        this.modelConfig = ModelConfig.getInstance(properties);
        this.sparkSession = SparkUtils.createSparkSession(modelConfig);
    }

    public void trainModel() {
        try {
            Dataset<Row> trainingData = sparkSession.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load(modelConfig.getTrainingDataPath());

            Pipeline pipeline = createPipeline(trainingData.schema());
            PipelineModel model = pipeline.fit(trainingData);

            CrossValidatorModel cvModel = trainCrossValidator(model, trainingData);

            evaluateModel(cvModel, trainingData);

            saveModel(cvModel, modelConfig.getModelPath());
        } catch (Exception e) {
            throw new ModelException("Error training model", e);
        }
    }

    private Pipeline createPipeline(org.apache.spark.sql.types.StructType schema) {
        // create pipeline stages based on configuration
        PipelineStage[] stages = new PipelineStage[modelConfig.getStages().size()];
        for (int i = 0; i < modelConfig.getStages().size(); i++) {
            stages[i] = modelConfig.getStages().get(i).createStage(schema);
        }
        return new Pipeline().setStages(stages);
    }

    private CrossValidatorModel trainCrossValidator(PipelineModel model, Dataset<Row> trainingData) {
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(model)
                .setEvaluator(new Evaluator())
                .setEstimatorParamMaps(new ParamMap[] { ParamMap.empty() })
                .setNumFolds(modelConfig.getCrossValidationFolds());

        return crossValidator.fit(trainingData);
    }

    private void evaluateModel(CrossValidatorModel model, Dataset<Row> testData) {
        // evaluate model using configured evaluator
        double evaluationMetric = modelConfig.getEvaluator().evaluate(model, testData);
        System.out.println("Evaluation metric: " + evaluationMetric);
    }

    private void saveModel(CrossValidatorModel model, String path) {
        model.write().overwrite().save(path);
        System.out.println("Model saved to: " + path);
    }

    public static void main(String[] args) {
        Properties properties = new Properties();
        properties.setProperty("model.config.path", "src/main/resources/model.properties");
        DataDrivenMachineLearningController controller = new DataDrivenMachineLearningController(properties);
        controller.trainModel();
    }
}