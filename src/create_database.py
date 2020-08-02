import os
import sys
import logging
import logging.config
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy import Column, Integer, String, MetaData, Float
import config.flaskconfig as config
import pandas as pd


Base = declarative_base()

class Insurance(Base):
	"""
	Create a data model for the database:

	age: age of primary beneficiary
	sex: insurance contractor gender, female, male
	bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
	objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
	children: Number of children covered by health insurance / Number of dependents
		smoker: Smoking
	region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
	charges: Individual medical costs billed by health insurance
	"""
	__tablename__ = 'insurance'
	id = Column(Integer, primary_key=True, nullable=False)
	age = Column(Integer, nullable=False)
	sex = Column(String(100), nullable=False)
	bmi = Column(Integer, nullable=False)
	children = Column(Integer, nullable=False)
	smoker = Column(String(100), nullable=False)
	region = Column(String(100), nullable=False)
	charges = Column(Float, nullable=True)


	def __repr__(self):
		return '<Insurance %r>' % self.title


def createDB():
	"""create database schema locally in sqlite or in RDS based on a configuration.'

	        Args:
	            args: subparser in run.py

	        Returns:
	            void
	"""
	# set up logging
	logging.config.fileConfig(config.LOGGING_CONFIG)
	logger = logging.getLogger(__file__)
	print(config.SQLALCHEMY_DATABASE_URI)
	# Create database locally if Local_DB is True
	if config.LOCAL_DB:
		logger.info("Creating db locally...")
		engine_string = config.LOCAL_SQLALCHEMY_DATABASE_URI
	# Otherwise create database on RDS
	else:
		logger.info("Creating db on RDS...")
		engine_string = config.SQLALCHEMY_DATABASE_URI


	# set up mysql connection
	try:
		engine = sql.create_engine(engine_string)
	except Exception as e:
		logger.error('Cannot establish database connection.', e)

	# create the tracks table
	try:
		Base.metadata.create_all(engine)
	except Exception as e:
		logger.error('Cannot create table schema.', e)
	else:
		logger.info("Successfully created table schema!")

	# create a db session
	Session = sessionmaker(bind=engine)
	session = Session()

	# add a record/track
	trackList = []
	df = pd.read_csv(config.DATA_PATH)
	for index, row in df.iterrows():
		trackList.append(Insurance(
			id=index,
			age=row['age'],
			sex=row['sex'],
			bmi=row['bmi'],
			children=row['children'],
			smoker=row['smoker'],
			region=row['region'],
			charges=row['charges']
		))

	session.add_all(trackList)

	session.commit()
	logger.info("Database created with columns records added.")

	session.close()

