from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation_pipreine import DaraTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_3_model_trainer_pipeline import ModelTrainingPipeline
from src.textSummarizer.pipeline.stage_4_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME='Data Ingestion stage'
try:
    logger.info(f"Stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME='Data Transformation stage'
try:
    logger.info(f"sateg {STAGE_NAME} initiated")
    data_ingestion_pipeline=DaraTransformationTrainingPipeline()
    data_ingestion_pipeline.initiate_data_transformation()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME='Model Trainer Stage'
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_trainer_pipeline=ModelTrainingPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f"stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME='Model Evaluation stage'
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation=ModelEvaluationTrainingPipeline()
   model_evaluation.initiate_model_evaluation()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e