import numpy as np

from .base import BaseLearningModel

class AON(BaseLearningModel):
    """
    All-or-nothing model. This model does not learn, but selects the action with the lowest cost expectation.
    
    Args:
        params (dict): A dictionary containing model parameters.
        initial_knowledge (list or array): Initial knowledge of cost expectations.
    Attributes:
        cost (np.ndarray): Agent's cost expectations for each option.
    """
    def __init__(self, params, initial_knowledge):
        super().__init__()
        self.cost = np.array(initial_knowledge, dtype=float)
        
    def act(self, state) -> int:
        """Selects the action with the lowest cost expectation.

        Args:
            state (Any): The current state of the environment (not used).
        Returns:
            action (int): The index of the selected action.
        """

        action = int(np.argmax(self.cost))
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