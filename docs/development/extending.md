# In opensynthetics/training_eval/metrics/__init__.py
from opensynthetics.training_eval.metrics.accuracy import accuracy_score, f1_score
from opensynthetics.training_eval.metrics.regression import mean_squared_error, mean_absolute_error
from opensynthetics.training_eval.metrics.ranking import ndcg_score, mean_reciprocal_rank

# Register metrics
METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "ndcg": ndcg_score,
    "mrr": mean_reciprocal_rank,
}

def get_metric(name: str):
    """Get a metric function by name.

    Args:
        name: Metric name

    Returns:
        function: Metric function

    Raises:
        ValueError: If metric not found
    """
    if name not in METRICS:
        raise ValueError(f"Metric not found: {name}")
    return METRICS[name]