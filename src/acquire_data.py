import sys
import logging.config
import pandas as pd
import boto3
import config.flaskconfig as config

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__file__)

def source_to_s3():
    """Download data from Github source and the upload it to s3 as 'data.csv'

        Args:
            args: subparser in run.py

        Returns:
            void
    """
    # Download data from the source
    try:
        logger.info("Fetching insurance data...")
        df = pd.read_csv(config.DATA_SOURCE_URL)
    except Exception as e:
        logger.error('Error occured while fetching tweet ids.', e)
        sys.exit(1)

    # store data locally
    df.to_csv(config.DATA_PATH, index=False)

    # upload data to s3
    try:
        logger.info('uploading data to s3...')
        s3 = boto3.client("s3", aws_access_key_id=config.AWS_KEY, aws_secret_access_key=config.AWS_SECRET_KEY)
        s3.upload_file(config.DATA_PATH, 'data.csv')
    except Exception as e:
        logger.error('Error occured while uploading data to s3.', e)
        sys.exit(1)
    else:
        logger.info('Successfully uploaded data to s3')
