from abc import ABC, abstractmethod

class BaseLearningModel(ABC):
    """
    This is an abstract base class for the baseline models.\n
    Users can create their own learning models by inheriting from this class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def act(self, state) -> None:
        """Method to select an action based on the current state and cost.

        Returns:
            None
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward) -> None:
        """Method to learn the model based on the current state and cost.

        Arguments:
            state (Any): The current state of the environment.
            action (Any): The action to take.
            reward (Any): The reward received from the environment.
        Returns:
            None
        """

        pass
