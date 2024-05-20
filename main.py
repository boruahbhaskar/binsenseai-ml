from src.binsenseai import logger
from src.binsenseai.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.binsenseai.pipeline.stage_01_data_transformation import DataReadingAndTransformationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Read Excel sheet"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataReadingAndTransformationPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Make train and validation split files"
STAGE_NAME = "Make metadata files metadata.json instances.json"