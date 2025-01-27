from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = 'Data Ingestion Stage'
try:
    logger.info(f'>>>>>>>>>> {STAGE_NAME} - Started <<<<<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>>>>> {STAGE_NAME} - Completed <<<<<<<<<<')
except Exception as e:
    logger.error(f'Error in {STAGE_NAME} - {str(e)}')
    raise e