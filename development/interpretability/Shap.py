import shap

from lime.lime_text import LimeTextExplainer
from .BaseInterpreter import BaseInterpreter
from torch import softmax

class Shap(BaseInterpreter):
    def __init__(self):
        self.__explainer = None

    def interpret(self, input_text, transformer, data):
        self.__explainer = shap.DeepExplainer(transformer.model, data)
        model_inputs = transformer.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

      
        shap_values = self.__explainer(model_inputs)
        shap.summary_plot(shap_values, model_inputs['input_ids'], tokenizer=transformer.tokenizer)

    def print_parameters(self):
        parameters = f'''
            Parameter Name - Data Type - Explaination

            1. input_text - string - input text for the model
            2. transformer - Object - hugging face transformer for text classification
        '''

        print(parameters)






