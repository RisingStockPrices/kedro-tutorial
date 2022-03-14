"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_movies

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_movies,
            inputs="movies",
            outputs="preprocessed_movies",
            name="preprocess_movies_node"
        ),
    ])
