"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from sklearn import linear_model
import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def train(data:pd.DataFrame, labels:pd.DataFrame):
    reg = linear_model.LinearRegression()
    import pdb;pdb.set_trace()
    reg.fit(data,labels)
    
    return reg
    
def split_data(data:pd.DataFrame, split_ratio:float):
    idx = int(len(data)*split_ratio)
    
    # df_test = movies[:idx]
    # df_train = movies[idx:]

    # df_test_x = df_test['revenue']
    # df_test_y = df_test['vote_average']

    # df_train_x = df_train['revenue']
    # df_train_y = df_train['vote_average']

    X = data[['revenue','budget']]#parameters["features"]]
    y = data["vote_average"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=idx#, random_state=parameters["random_state"]
    )
    # return X_train, X_test, y_train, y_test
    return {
        "train_x":X_train,#df_train_x,
        "train_y":y_train,#df_train_y,
        "test_x":X_test,#df_test_x,
        "test_y":y_test#df_test_y
    }
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    # import pdb;pdb.set_trace()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)