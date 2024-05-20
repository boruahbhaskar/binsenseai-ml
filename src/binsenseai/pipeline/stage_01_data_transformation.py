from src.binsenseai.config.configuration import ConfigurationManager
from src.binsenseai.utils.load_data_S3 import DataReadAndTransformation
from src.binsenseai import logger



STAGE_NAME = "Data Ingestion stage"
class DataReadingAndTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_transformation_config()
        data_ingestion = DataReadAndTransformation(config=data_ingestion_config)
        
        #data_list = data_ingestion.read_excel_to_list("artifacts/data_ingestion/binsimages.xlsx","Sheet1") #binsimages
        #data_ingestion.convert_to_string_with_zeros(5) # 5 number width
        #data_ingestion.s3_read_write(data_list) # 5 number width
        
        
        
        #data_ingestion.make_metadata('artifacts/data_ingestion/amazon_bin_images','artifacts/data_ingestion/amazon_bin_metadata') # 5 number width 
        #data_ingestion.split_train_val_data('artifacts/data_ingestion/amazon_bin_images','artifacts/data_ingestion/amazon_bin_metadata')
        
        #data_ingestion.make_obj_num_verification_data('artifacts/data_transformation/random_train.txt',
        #                                              'artifacts/data_transformation/random_val.txt',
        #                                              'artifacts/data_transformation/metadata.json',
        #                                              'artifacts/data_transformation/instances.json'
        #                                              )
        # data_ingestion.make_obj_verification_data('artifacts/data_transformation/random_train.txt',
        #                                               'artifacts/data_transformation/random_val.txt',
        #                                               'artifacts/data_transformation/metadata.json',
        #                                               'artifacts/data_transformation/instances.json'
        #                                               )
        

        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataReadingAndTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e