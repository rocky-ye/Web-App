import pandas as pd
import numpy as np
import pickle
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import xgboost

from src.generate_features import choose_features, get_target, generate_features
from src.train_model import split_data, train_model
from src.evaluate_model import evaluate_model
from src.score_model import score_model

def test_choose_features():
    """Test the functionality of choose_features."""
    # load sample test data
    data = pd.read_csv("test/test_data.csv")

    features = ['CreditScore', 'Tenure', 'NumOfProducts', 'HasCrCard']

    # desired output dataframe
    output_df = data[['CreditScore', 'Tenure', 'NumOfProducts', 'HasCrCard', 'Exited']]
    
    # raise AssertionError if dataframes do not match
    assert output_df.equals(choose_features(df=data, features_to_use=features, target='Exited'))


def test_get_target():
    """Test the functionality of get_target."""
    # load sample test data
    data = pd.read_csv("test/test_data.csv")

    # desired output values
    output_values = data['Exited'].values
    
    # raise AssertionError if output values do not match element-wise
    assert (output_values==(get_target(df=data, target='Exited'))).all()


def test_target_name():
    """Test the get_target script handles invalid columns as expected."""
    with pytest.raises(KeyError) as excinfo:
        # load sample test data
        data = pd.read_csv("test/test_data.csv")
        # unnamed_col is not a column of test_data
        get_target(df=data, target='unnamed_col')
    # raise AssertionError if error message is not as expected
    # note: KeyError message is wrapped by additional ""
    assert str(excinfo.value) == "'Not a valid column of this data!'"


def test_generate_features():
    """Test the functionality of generate_features."""
    # load sample test data
    data = pd.read_csv("test/test_raw_data.csv")
    kwargs = {'choose_features':{'features_to_use': ['CreditScore','Tenure','NumOfProducts','Gender','Geography'],
                                 'target': 'Exited'},
            'to_dummy':['Gender','Geography']}
    # output dataframe
    output = data[['CreditScore','Tenure','NumOfProducts','Gender','Geography','Exited']]
    # convert two variables to dummies
    gender_dummy = pd.get_dummies(output['Gender'], drop_first=True)
    geo_dummy = pd.get_dummies(output['Geography'], drop_first=True)

    output.drop(['Gender','Geography'],axis=1,inplace=True)
    # new dataframe after encode categorical variables as dummies
    output = pd.concat([output,gender_dummy,geo_dummy], axis=1)

    assert output.equals(generate_features(df=data, save_features=None, **kwargs))


def test_split_data():
    """Test the functionality of split_data."""
    # load sample test data
    data = pd.read_csv("test/test_data.csv")
    X_df = data.drop('Exited',axis=1)
    y_df = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=123)

    # split data using the function
    X, y = split_data(X_df, y_df, train_size=0.7, test_size=0.3, random_state=123)

    # raise AssertionError if keys do not match
    assert X_train.equals(X['train'])
    assert y_test.equals(y['test'])


def test_model_input():
    """Test whether train_model script for xgboost handles non-numeric input as expected."""
    # train_model will raise ValueError if a dataframe with object or string column is used as input
    # error message is specified in train_model script
    with pytest.raises(ValueError) as excinfo:
        # load sample test data
        data = pd.read_csv("test/test_raw_data.csv")
        methods = dict(xgboost=xgboost.XGBClassifier)
        # model parameters
        max_depth = 3
        n_estimators = 300
        learning_rate = 0.05
        # model to train
        method = 'xgboost'
        # predefined arguments
        model_kwargs = {'choose_features':{'features_to_use': ['CreditScore','Tenure','NumOfProducts','Gender','Geography']},
                  'get_target':{'target':'Exited'},
                  'split_data':{'train_size':0.7, 'test_size':0.3, 'random_state':42}}
        # input column contains strings so expected to throw ValueError during model fit
        train_model(df=data, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, method=method, save_tmo=None, **model_kwargs)
    # raise AssertionError if error message is not as expected
    assert str(excinfo.value) == 'Input dataframe can only have numeric or boolean types!'


def test_model_type():
    """Test whether the trained model created from train_model script is of class xgboost."""
    # load sample test data
    data = pd.read_csv("test/test_data.csv")
    methods = dict(xgboost=xgboost.XGBClassifier)
    # model parameters
    max_depth = 3
    n_estimators = 300
    learning_rate = 0.05
    # model to train
    method = 'xgboost'
    # predefined arguments
    model_kwargs = {'choose_features':{'features_to_use': ['CreditScore','Tenure','NumOfProducts','Germany','Male']},
                'get_target':{'target':'Exited'},
                'split_data':{'train_size':0.7, 'test_size':0.3, 'random_state':42}}
    # input column contains strings so expected to throw ValueError during model fit
    xgb_bin = train_model(df=data, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, method=method, save_tmo=None, **model_kwargs)
    # check if the model type is right
    expected_type = "<class 'xgboost.sklearn.XGBClassifier'>"
    assert str(type(xgb_bin)) == expected_type


def test_model_output():
    """Test whether the trained model from train_model script is the expected model with certain attributes."""
        # load sample test data
    data = pd.read_csv("test/test_data.csv")
    methods = dict(xgboost=xgboost.XGBClassifier)
    # model parameters
    max_depth = 3
    n_estimators = 300
    learning_rate = 0.05
    # model to train
    method = 'xgboost'
    # predefined arguments
    model_kwargs = {'choose_features':{'features_to_use': ['CreditScore','Tenure','NumOfProducts','Germany','Male']},
                'get_target':{'target':'Exited'},
                'split_data':{'train_size':0.7, 'test_size':0.3, 'random_state':42}}
    # input column contains strings so expected to throw ValueError during model fit
    xgb_bin = train_model(df=data, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42, method=method, save_tmo=None, **model_kwargs)

    # expected model
    X_df = data[['CreditScore','Tenure','NumOfProducts','Germany','Male']]
    y_df = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42)
    # initialize the xgboost classifier
    xgb_classifier = xgboost.XGBClassifier(objective='binary:logistic', learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, random_state=42)  
    # fit the classifier on training data
    xgb_classifier.fit(X_train,y_train)

    # check if the model is the same as expected using same random seed of 42
    assert str(xgb_classifier.get_xgb_params) == str(xgb_bin.get_xgb_params)
    # check if the model has the attributes that imply it has been trained - in this case feature importances
    assert xgb_bin.feature_importances_ is not np.nan


def test_score_predict():
    """Test whether the unfitted model used in score_model is handled as expected."""
    # initialize the xgboost classifier
    xgb_classifier = xgboost.XGBClassifier()
    
    with pytest.raises(NotFittedError) as excinfo:
        path_to_tmo = 'test/empty.pkl'
        # save the logmodel in test folder
        with open(path_to_tmo, "wb") as f:
            pickle.dump(xgb_classifier, f)
        # load sample test data
        data = pd.read_csv("test/test_data.csv")
        score_kwargs = {'none':'none'}
        X_data = data[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Male']]
        score_model(df=X_data, path_to_tmo=path_to_tmo, threshold=0.5, **score_kwargs)
    # raise AssertionError if error message is not as expected
    assert str(excinfo.value) == 'Model needs to be fitted before making predictions!'


def test_score_model():
    """Test the functionality of score_model."""
    path_to_tmo = 'test/test-model.pkl'
    # load sample test data
    data = pd.read_csv("test/test_data.csv")
    kwargs = {'none':'none'}
    X_data = data[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Male']]

    with open(path_to_tmo, "rb") as f:
        model = pickle.load(f)
    # get probability of churn
    y_prob = model.predict_proba(X_data)[:,1]
    output = pd.DataFrame(y_prob)
    output.columns = ['pred_prob']

    # assign class label based on threshold
    output['pred'] = output.apply(lambda row: 1 if row['pred_prob']>0.6 else 0, axis=1)
    
    assert output.equals(score_model(df=X_data, path_to_tmo=path_to_tmo, threshold=0.6, **kwargs))


def test_score_input():
    """Test the input features to be scored by xgboost model are all numeric or boolean."""
    # score_model will raise ValueError if a dataframe with object or string column is used as input
    # error message is specified in score_model script
    with pytest.raises(ValueError) as excinfo:
        path_to_tmo = 'test/test-model.pkl'
        # load sample test data
        data = pd.read_csv("test/test_raw_data.csv")
        kwargs = {'none':'none'}
        X_data = data[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Geography', 'Gender']]
        score_model(df=X_data, path_to_tmo=path_to_tmo, threshold=0.6, **kwargs)
    # raise AssertionError if error message is not as expected
    assert str(excinfo.value) == 'Input dataframe can only have numeric or boolean types!'


def test_score_prob():
    """Test the scored probabilities are in the range of 0-1 and predicted classes are either 1 or 0."""
    path_to_tmo = 'test/test-model.pkl'
    # load sample test data
    data = pd.read_csv("test/test_data.csv")
    kwargs = {'none':'none'}
    X_data = data[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Germany', 'Male']]
    # get predicted class and probability
    prob_scores = score_model(df=X_data, path_to_tmo=path_to_tmo, threshold=0.5, **kwargs).iloc[:,0]
    class_scores = score_model(df=X_data, path_to_tmo=path_to_tmo, threshold=0.5, **kwargs).iloc[:,1]
    # all of predicted prob have to be in range 0-1 inclusive
    assert prob_scores.between(0,1,inclusive=True).all()
    # all of predicted classes have to be either 0 or 1
    assert class_scores.isin([0,1]).all()


def test_score_type():
    """Test whether the model scoring script handles wrong model type as expected."""
    data_dic = {'class':[0,1,1,0,0],
            'feature1':[2,5,7,8,2],
            'feature2':[3,6,8,4,5],
            'feature3':[10,15,10,12,11]}
    data = pd.DataFrame(data_dic)
    # use sample data to build a logistic regression
    logmodel = LogisticRegression()
    logmodel.fit(data[['feature1','feature2','feature3']],data['class'])

    save_tmo = 'test/test-logit.pkl'
    # save the logmodel in test folder
    with open(save_tmo, "wb") as f:
        pickle.dump(logmodel, f)

    # score_model will raise TypeError if a logistic regression model is used as input
    # error message is specified in score_model script
    with pytest.raises(TypeError) as excinfo:
        kwargs = {'none':'none'}
        score_model(df=data[['feature1','feature2','feature3']], path_to_tmo=save_tmo, threshold=0.6, **kwargs)
    # raise AssertionError if error message is not as expected
    assert str(excinfo.value) == "model used to score must be an XGBoost Classifier"


def test_evaluate_model():
    """Test the functionality of evaluate_model."""
    # test data input
    score_input = {'pred_prob': [0.998,0,0.99,0.914,0.993,0,0.006,0.999,0.00046,0.999],
                   'pred': [1,0,1,1,1,0,0,1,0,1]}
    label_input = {'class':[0,1,0,1,0,1,0,0,1,0]}

    score_df = pd.DataFrame(score_input)
    label_df = pd.DataFrame(label_input)

    # desired output dataframe
    output = confusion_matrix(label_df, score_df.iloc[:,1])
    output_df = pd.DataFrame(output,
        index=['Actual Negative','Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive'])
    
    # add kwargs for function
    pre_defined_kwargs = {'metrics':["confusion_matrix"]}
    # raise AssertionError if dataframes do not match
    assert output_df.equals(evaluate_model(label_df, score_df, **pre_defined_kwargs))


def test_evaluate_indexing():
    """Test the evaluate_model script handles index out of bounds as expected."""
    with pytest.raises(IndexError) as excinfo:
        # test data input
        score_input = {'pred_prob': [0.998,0,0.99,0.914,0.993,0,0.006,0.999,0.00046,0.999]}
        label_input = {'class':[0,1,0,1,0,1,0,0,1,0]}
        score_df = pd.DataFrame(score_input)
        label_df = pd.DataFrame(label_input)
        pre_defined_kwargs = {'metrics':["confusion_matrix"]}
        evaluate_model(label_df, score_df, **pre_defined_kwargs)
    # raise AssertionError if error message is not as expected
    assert str(excinfo.value) == 'Index out of bounds!'


def test_evaluate_inputs():
    """Test the evaluate_model script handles invalid inputs as expected."""
    with pytest.raises(ValueError) as excinfo1:
        # test data input
        score_input = {'pred_prob': ['0.998','0','0.99','0.914','0.993','0','0.006','0.999','0.00046','0.999'],
                       'pred': [1,0,1,1,1,0,0,1,0,1]}
        label_input = {'class':[0,1,0,1,0,1,0,0,1,0]}
        score_df = pd.DataFrame(score_input)
        label_df = pd.DataFrame(label_input)
        pre_defined_kwargs = {'metrics':["confusion_matrix"]}
        evaluate_model(label_df, score_df, **pre_defined_kwargs)

    with pytest.raises(ValueError) as excinfo2:
        # test data input
        score_input2 = {'pred_prob': [0.998,0,0.99,-1,0.993,0,0.006,0.999,0.00046,3],
                        'pred': [1,0,1,1,1,0,0,1,0,1]}
        label_input2 = {'class':[0,1,0,1,0,1,0,0,1,0]}
        score_df2 = pd.DataFrame(score_input2)
        label_df2 = pd.DataFrame(label_input2)
        pre_defined_kwargs2 = {'metrics':["confusion_matrix"]}
        evaluate_model(label_df2, score_df2, **pre_defined_kwargs2)

    # raise AssertionError if error message is not as expected
    assert str(excinfo1.value) == 'Input dataframe can only have numeric or boolean types!'
    assert str(excinfo2.value) == 'Probabilities needs to be in 0-1 range!'





