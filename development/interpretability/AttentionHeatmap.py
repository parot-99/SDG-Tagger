import matplotlib.pyplot as plt
import seaborn as sns
from .BaseInterpreter import BaseInterpreter
from torch import argmax


class AttentionHeatmap(BaseInterpreter):
    def interpret(self, input_text, transformer, attention_head_id=0):
        self.__model_inputs = transformer.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        outputs = transformer.model(**self.__model_inputs)
        self.__predicted_class = argmax(outputs.logits)
        self.__attention_scores = outputs.attentions[0][0][attention_head_id]
        self.__visualize(transformer)

    def __visualize(self, transformer):
        plt.figure(figsize=(25, 25))
        sns.heatmap(
            self.__attention_scores.detach().numpy(),
            cmap="YlGnBu",
            xticklabels=transformer.tokenizer.convert_ids_to_tokens(self.__model_inputs['input_ids'][0]),
            yticklabels=transformer.tokenizer.convert_ids_to_tokens(self.__model_inputs['input_ids'][0])
        )
        class_id = self.__predicted_class.item()
        plt.title(
            f'Heatmap of Attention Scores for Class {class_id}'
        )
        plt.show()

    def print_parameters(self):
        parameters = f'''
            Parameter Name - Data Type - Explaination

            1. input_text - string - input text for the model
            2. transformer - Object - hugging face transformer for text classification
            3. attention_head_id - int - Id of attention head to visualize
        '''

        print(parameters)


