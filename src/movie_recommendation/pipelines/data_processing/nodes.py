"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

import pandas as pd
from typing import Any,Dict
def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:

    return movies[["revenue","vote_average","budget"]]#[["original_title","budget","genres","revenue","vote_average"]]
