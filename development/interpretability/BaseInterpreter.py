from abc import ABC, abstractmethod


class BaseInterpreter(ABC):
    """ Base Interpreter abstract class that each interpreter in this module must inherit from, it's used to assure that any interpreter added has the interpret and print_parameters method. The methods are used in the Interpreter class to run interpretations and print information about a specific interpreter's parameters
    
    """
    @abstractmethod
    def interpret():
        pass

    @abstractmethod
    def print_parameters():
        pass