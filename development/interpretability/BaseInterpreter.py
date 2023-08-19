from abc import ABC, abstractmethod


class BaseInterpreter(ABC):
    @abstractmethod
    def interpret():
        pass

    @abstractmethod
    def print_parameters():
        pass