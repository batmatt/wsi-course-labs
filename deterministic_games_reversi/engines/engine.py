from abc import abstractmethod


class IEngine:
    @abstractmethod
    def perform_move(self, possible_moves):
        raise NotImplementedError
