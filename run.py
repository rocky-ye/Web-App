import argparse
from src.create_database import *
from src.acquire_data import source_to_s3
from src.modeling import modeling
import config.flaskconfig as config
logging.basicConfig(format='%(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('__name__')

if __name__ == '__main__':

    # collect user input using parser
    parser = argparse.ArgumentParser(description="Running the pipeline")

    parser.add_argument('step', help='Which step to run', choices=['acquire_data', 'create_db', 'run_model', 'test'])
    parser.add_argument('--local', '-i', action='store_true', help='connect to local database')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', '-o', default=None, help='Path to save output CSV (optional, default = None)')

    args = parser.parse_args()

    # connect to local or RDS data base
    if args.local:
        config.LOCAL_DB = True

        logger.info('Connecting to local database.')
    # run script according to user input
    if args.step == 'acquire_data':
        source_to_s3()
    elif args.step == 'create_db':
        createDB()
    elif args.step == 'run_model':
        modeling()