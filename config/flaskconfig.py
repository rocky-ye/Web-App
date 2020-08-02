import os
from os import path
DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000
APP_NAME = "insurance charge predictor"
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 3
PROJECT_HOME = path.dirname(path.dirname(path.abspath(__file__)))
LOGGING_CONFIG = path.join(PROJECT_HOME, 'config/logging/local.conf')
RESULT = ''
# Data Source
DATA_SOURCE_URL = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'

# s3 Connection
AWS_KEY = os.environ.get('AWS_KEY')  # Please set up your AWS_Key
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')  # Please set up your AWS_SECRET_Key
AWS_BUCKET_NAME = 'msia-rocky'  # Can be configured to other buckets

# Local Data and Database Path
DATA_PATH = 'data/data.csv'
LOCAL_DB = False


# RDS Connection
DB_HOST = os.environ.get('MYSQL_HOST')
DB_PORT = os.environ.get('MYSQL_PORT')
DB_USER = os.environ.get('MYSQL_USER')
DB_PW = os.environ.get('MYSQL_PASSWORD')
DATABASE = os.environ.get('DATABASE_NAME')
DB_DIALECT = 'mysql+pymysql'
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')


LOCAL_SQLALCHEMY_DATABASE_URI = 'sqlite:///data/insurance.db'

SQLALCHEMY_DATABASE_URI = '{dialect}://{user}:{pw}@{host}:{port}/{db}'.format(dialect=DB_DIALECT, user=DB_USER,
                                                                                  pw=DB_PW, host=DB_HOST, port=DB_PORT,
                                                                                  db=DATABASE)
