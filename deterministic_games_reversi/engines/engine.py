from abc import abstractmethod


class IEngine:
    """
    Abstract class representing the engine with its own logic of making moves

    """

    @abstractmethod
    def perform_move(self, possible_moves: dict):
        """
        Makes move based on implemented engine logic

        """
        raise NotImplementedError
