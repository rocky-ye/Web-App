import pytest
import pandas as pd
import numpy as np
import src.predict
import src.modeling

def test_predicting_happy():
    df = pd.DataFrame([[19, 'female', 27.9, 0, 'yes', 'southwest']],
                        columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    true_value = 17343.64800266597
    assert abs(src.predict.predicting(df) - true_value) < 0.01

def test_predicting_unhappy():
    df = None
    true_value = None
    with pytest.raises(TypeError):
        src.predict.predicting(df)

def test_one_hot_encoding_happy():
    df = pd.DataFrame([['a', 'b', 1],
                       ['c', 'd', 2]], columns=['fir', 'sec', 'thir'])
    enc, column = fitted = src.modeling.one_hot_encoding_fit(df, ['fir', 'sec'], ['thir'])
    transformed = src.modeling.one_hot_encoding_transform(enc, df, ['fir', 'sec'], ['thir'])
    true = np.array([[1, 1, 0, 1, 0], [2, 0, 1, 0, 1]])
    assert np.array_equal(transformed, true)

def test_one_hot_encoding_unhappy():
    df = None
    true = None
    assert src.modeling.one_hot_encoding_transform(None, None, None, None) == true