from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = 'Data Ingestion Stage'
try:
    logger.info(f'>>>>>>>>>> {STAGE_NAME} - Started <<<<<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>>>>> {STAGE_NAME} - Completed <<<<<<<<<<')
except Exception as e:
    logger.error(f'Error in {STAGE_NAME} - {str(e)}')
    raise e





STAGE_NAME = 'Prepare Base Model Stage'
try:
    logger.info(f'>>>>>>>>>> {STAGE_NAME} - Started <<<<<<<<<<')
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f'>>>>>>>>>> {STAGE_NAME} - Completed <<<<<<<<<<')
except Exception as e:
    logger.exception(e)
    raise e