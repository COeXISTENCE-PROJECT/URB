from routerl import get_learning_model

from .aon import AON
from .random import Random

def get_baseline(params, initial_knowledge):
    """Returns a learning model based on the provided parameters.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (Any): A dictionary containing initial knowledge.
    Returns:
        BaseLearningModel: A learning model object.
    Raises:
        ValueError: If model is unknown.
    """

    model = params["model"]
    if model == "aon":
        return AON(params, initial_knowledge)
    elif model == "random":
        return Random(params, initial_knowledge)
    else:
        try:
            return get_learning_model(params, initial_knowledge)
        except:
            raise ValueError('[MODEL INVALID] Unrecognized model: ' + model)