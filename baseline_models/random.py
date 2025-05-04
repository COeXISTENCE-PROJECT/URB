import numpy as np
import random

from .base import BaseLearningModel

class Random(BaseLearningModel):
    """
    Random model. This model selects actions randomly without any learning or cost expectations.

    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    Attributes:
        cost (np.ndarray): Agent's cost expectations for each option.
    """

    def __init__(self, params, initial_knowledge):
        self.cost = np.array(initial_knowledge, dtype=float)
        
    def act(self, state) -> int:
        """Selects an action randomly.

        Args:
            state (Any): The current state of the environment (not used).
        Returns:
            action (int): The index of the selected action.
        """

        action = random.randint(0, len(self.cost) - 1)
        return action
    
    def learn(self, state, action, reward) -> None:
        """Does not learn or update any cost expectations.

        Args:
            state (Any): The current state of the environment (not used).
            action (int): The action that was taken (not used).
            reward (float): The reward received after taking the action (not used).
        Returns:
            None
        """
        pass