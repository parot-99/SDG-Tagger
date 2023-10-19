from .AttentionHeatmap import AttentionHeatmap
from .Lime import Lime
from .Shap import Shap

class Interpreter:
    """ The Interpreter class provides a collection of interpretation frameworks that are used to provide explainations for the TextClassifiers provided in this work. Each Interpretation framework is provided by its own class that inherets from BaseInterpreter.

    To add a new Interpreter, provide it as class, implement print_interpreter_parameters and interpret methods, and add it to the interpreters list in this class.

    Attributes:
    -----------
    interpreter_type: object
        sets the interpreter to be used

    Methods:
    --------
    print_interpreter_names:
        prints the list of available interpreters
    print_interpreter_parameters:
        prints configurable list of parameters for the desired interpreter
    interpret:
        provides explainations for the desired model
    
    """
    
    def __init__(self, interpreter_type):
        self.__interpreters = {
            'attention_heatmap': AttentionHeatmap,
            'lime': Lime,
        }
        self.__interpreter = None

        if interpreter_type:
            self.__interpreter = self.__interpreters[interpreter_type]()


    def print_interpreter_names(self):
        for i, interpreter in enumerate(self.__interpreters.keys()):
            print(f'{i + 1}. {interpreter}')

    def print_interpreter_parameters(self):
        if self.__interpreter:
            self.__interpreter.print_parameters()

            return

        raise Exception(
            'Interpreter not set yet, use `print_interpreter_names` to see available interpreters'
        )

    def interpret(self, **kwargs):
        if self.__interpreter:
            self.__interpreter.interpret(**kwargs)

            return

        raise Exception(
            'Interpreter not set yet, use `print_interpreter_names` to see available interpreters'
        )


    # properties

    @property
    def interpreter(self):
        if self.__interpreter:
            return self.__interpreter
        
        raise Exception(
            'Interpreter not set yet, use `print_interpreter_names` to see available interpreters'
        )

    @interpreter.setter
    def interpreter(self, interpreter_type):
        if interpreter_type not in self.__interpreters.keys():
            raise Exception(
                'Interpreter does not exist, use `print_interpreter_names` to see available interpreters'
            )
        
        self.__interpreter = self.__interpreters[interpreter_type]()
