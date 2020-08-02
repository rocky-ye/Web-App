# Web App

### A full stack web development project utilizing AWS S3 & RDS for cloud backend storage, a python machine learning pipeline for modeling, Flask and CSS for the front end web page.

<!-- toc -->
- [Project Charger](#Project-charter)
- [Directory Structure](#directory-structure)
- [Running the App in Docker](#running-the-App-in-Docker)
  * [Acquire Data](#1-Acquire-Data)
  * [Create Database](#2-Create-Database)
  * [Modeling](#3-Modeling)
  * [Unit Testing](#4-Unit-Testing)
  * [Running the App](#5-Running-the-app)
- [Backlog](#Backlog)


<!-- tocstop -->
# Project Charter 
### Vision:
Help health insurance buyers understand how much they can expect to pay and how each feature contributes to their final price. With this tools, users can get an idea if they are overpaying for their health insurance and how the price is determined.

### Mission:
Data Source: https://www.kaggle.com/mirichoi0218/insurance

We will use features in the above dataset to compare how different people pay different health insurance prices. We will then use machine learning models to predict that target value of future observations. Finally we will create an easy to use interactive web interface to carry out the functionality of the model and provide the user with an estimate of their insurance charge.

### Success Criteria:
The success criteria for this project will be an R2 metric greater than 80% tested on unseen datasets. The business outcome for this app is the app users getting the best insurance prices out there and gaining an understanding of what they can do to lower their price in the future. We need to track both the model accuracy and user happiness rate (can be estimated using survey and traffic).


# Directory Structure 

```
├── README.md                         <- You are here
├── api
│   ├── static/                       <- CSS, JS files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs
│   ├── boot.sh                       <- Start up script for launching app in Docker container.
│   ├── Dockerfile                    <- Dockerfile for building image to run app  
│
├── config                            <- Directory for configuration files 
│   ├── local/                        <- Directory for keeping environment variables and other local configurations that *do not sync** to Github 
│   ├── logging/                      <- Configuration of python loggers
│   ├── flaskconfig.py                <- Configurations for Flask API 
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── external/                     <- External data sources, usually reference data,  will be synced with git
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│
├── deliverables/                     <- Any white papers, presentations, final work products that are presented or delivered to a stakeholder 
│
├── docs/                             <- Sphinx documentation based on Python docstrings. Optional for this project. 
│
├── figures/                          <- Generated graphics and figures to be used in reporting, documentation, etc
│
├── models/                           <- Trained model objects (TMOs), model predictions, and/or model summaries
│
├── notebooks/
│   ├── archive/                      <- Develop notebooks no longer being used.
│   ├── deliver/                      <- Notebooks shared with others / in final state
│   ├── develop/                      <- Current notebooks being used in development.
│   ├── template.ipynb                <- Template notebook for analysis with useful imports, helper functions, and SQLAlchemy setup. 
│
├── reference/                        <- Any reference material relevant to the project
│
├── src/                              <- Source data for the project 
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── app.py                            <- Flask wrapper for running the model 
├── run.py                            <- Simplifies the execution of one or more of the src scripts  
├── requirements.txt                  <- Python package dependencies 
```
# Running the App in Docker

### 1. Acquire Data

#### Running script to acquire data from data source and put it into a configurable S3 bucket:

1. Go to `2020-msia423-Ye/config/flaskconfig.py` and edit the following variables to values you want to use:

    `AWS_KEY `

    `AWS_SECRET_KEY`

    `AWS_BUCKET_NAME`
    
2. Change to the `2020-msia423-Ye` directory 
    ```bash
    cd 2020-msia423-Ye
    ```
3. Build the insurance docker image 
    ```bash
    docker build -t insurance .
    ```     
4. Run the docker container acquire_data function using the insurance docker image 
    ```bash
    docker run insurance run.py acquire_data
    ```

### 2: Create Database

#### Running script to creates database schema locally in sqlite or in RDS based on a configuration:

#### Create database on RDS

1. Set up the following environment variable by running
    ```bash
    export MYSQL_USER=<MYSQL_USER>
    export MYSQL_PASSWORD=<MYSQL_PASSWORD>
    export MYSQL_HOST=msia-rocky.cpg3jjrdukfb.us-east-2.rds.amazonaws.com
    export MYSQL_PORT=3306
    export DATABASE_NAME=msia423
    ```
2. Using the same insurance docker image, run the docker container create_db function
    ```bash
    docker run -it \
    --env MYSQL_HOST \
    --env MYSQL_PORT \
    --env MYSQL_USER \
    --env MYSQL_PASSWORD \
    --env DATABASE_NAME \
    insurance run.py create_db
    ```
    
#### Create database locally
  
Using the same insurance docker image, run the docker container create_db function with option --local
    ```bash
    docker run --mount type=bind,source="$(pwd)"/data,target=/app/data insurance run.py create_db --local
    ```

### 3: Modeling:

#### Run the pipe line

1. set up s3 credentials:

```bash
export AWS_KEY=<your AWS_KEY>
export AWS_SECRET_KEY=<your AWS_SECRET_KEY>
```

2. build the same insurance docker image as previously if not built already.

```bash
docker build -t insurance .
```

3. run the following: (the script performs data processing and modeling. Data will be saved to the data folder and models will be saved to the model folder.)

```bash
docker run  --env AWS_KEY --env AWS_SECRET_KEY --mount type=bind,source="$(pwd)",target=/app insurance sh model_pipeline.sh
```

 ### 4: Unit Testing
### Perform unit tests with one happy path and one un happy path for each function

```bash
docker run --mount type=bind,source="$(pwd)",target=/app insurance sh unit_test.sh
```

### 5. Running the App

#### Configure Flask app 

`config/flaskconfig.py` holds the configurations for the Flask app. It includes the following configurations:

```python
DEBUG = True  # Keep True for debugging, change to False when moving to production 
LOGGING_CONFIG = "config/logging/local.conf"  # Path to file that configures Python logger
HOST = "0.0.0.0" # the host that is running the app. 0.0.0.0 when running locally 
PORT = 5000  # What port to expose app on. Must be the same as the port exposed in app/Dockerfile 
SQLALCHEMY_DATABASE_URI = 'sqlite:///data/tracks.db'  # URI (engine string) for database that contains tracks
APP_NAME = "penny-lane"
SQLALCHEMY_TRACK_MODIFICATIONS = True 
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100 # Limits the number of rows returned from the database 
```

#### Run the Flask app 

To run the Flask app using local database, run: 

```bash
docker run -p 5000:5000 --mount type=bind,source="$(pwd)",target=/app insurance python3 app.py --local
```

To run the Flask app using RDS, first set up all the environment variables (see Create database on RDS section above). Run: 

```bash
docker run -p 5000:5000 -it \
    --env MYSQL_HOST \
    --env MYSQL_PORT \
    --env MYSQL_USER \
    --env MYSQL_PASSWORD \
    --env DATABASE_NAME \
    insurance python3 app.py
```

You should now be able to access the app at http://0.0.0.0:5000/ in your browser.


#### Kill the container 

Once finished with the app, you will need to kill the container. To do so: 

```bash
(Mac) command + C
```


## Backlog
### Develop initiatives:
Allow insurance buyers understand how much they can expect to pay and how each feature contributes to their final price by using the tool.

### Broken down into epics:
**Epic 1：** Explore the data set to understand the relationship between the features and target
* story 1: Understand what each attribute represent and study overall sumary statistics and distribution of all variable across all observations.
* story 2: Clean the data set such as deleting duplicats, imputing missing values, finding outliers etc.
* story 3: Conduct exploratory analysis such as visulizing the correlation between each feature and the target. Try ansering questions like which state has the highest average insurance cost; how does age affect insurance cost etc.

**Epic 2:** Predict insurance prices with values from new users
* story 1: Use feature engineering to drop unrelated and uncorrelated features and create other features from the data to improve model performance.
* story 2: Build initial crude models to get a sense of predictive power of the data set and understand feature importance and what is going to be the next step.
* story 3: Fine tune the model and find the best model and parameters using cross validation.
* story 4: Write a pipeline to automate data cleaning and modeling.

**Epic 3:** Deploy the model onto AWS and develop the churn prediction App. 
* story 1: Push the model to the AWS

To be continued...


**Epic 4:** Keep track of App performance and evaluate the App with future data and user happiness scores.
* story 1: Check model performance every month and adjust the project based on the evaluation and feedback from users.

