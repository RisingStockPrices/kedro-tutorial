"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data,train_model,evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["preprocessed_movies","params:split_ratio"],
            outputs=dict(
                train_x="train_x",
                train_y="train_y",
                test_x="test_x",
                test_y="test_y"
            )
        ),
        node(
                func=train_model,
                inputs=["train_x", "train_y"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "test_x", "test_y"],
                outputs=None,
                name="evaluate_model_node",
            ),
    ])
