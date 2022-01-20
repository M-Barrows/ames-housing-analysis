from loguniform_int import loguniform_int
import my_module
import pytest
import pandas as pd

@pytest.mark.parametrize(
    'arg1,arg2',
    [
        (2,30),
        (1,100)
    ]
)
def test_loguniform_int_args(arg1,arg2):
    lu = loguniform_int(2, 30)
    assert lu._distribution.args == (2, 30)
    assert lu._distribution.args == (arg1,arg2)

@pytest.mark.parametrize(
    'csv,target',
    [
        ('../data/adult-census.csv', 'class'),
        ('../data/ames.csv', 'Sale_Price')
    ]
)
def test_return_types(csv, target):
    features, target = my_module.get_features_and_target(
        csv_file=csv,
        target_col=target
    )
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)

def test_return_types_census():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    
def test_return_types_ames():
    features, target = my_module.get_features_and_target(
        csv_file='../data/ames.csv',
        target_col='Sale_Price'
    )
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)

def test_cols_make_sense():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )
    # Load the data ourselves so we can double-check the columns
    df = pd.read_csv('../data/adult-census.csv')
    assert target.name in df.columns
    # Use a list comprehension to check all the feature columns
    assert all([feature_col in df.columns for feature_col in features])