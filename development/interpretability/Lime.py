from lime.lime_text import LimeTextExplainer
from .BaseInterpreter import BaseInterpreter
from torch import softmax

class Lime(BaseInterpreter):
    def __init__(self):
        self.__class_names = [
            'No Poverty',
            'Zero Hunger',
            'Good Health and Well-Being',
            'Quality Education',
            'Gender Equality',
            'Clean Water and Sanitation',
            'Affordable and Clean Energy',
            'Decent Work and Economic Growth',
            'Industry, Innovation, and Infrastructure',
            'Reduced Inequalities',
            'Sustainable Cites and Communities',
            'Responsible Consumption and Production',
            'Climate Action',
            'Life Below Water',
            'Life on Land',
            'Peace, Justice, and Strong Institutions'
        ]

    def interpret(self, input_text, transformer, show=True, device=0):
        explainer = LimeTextExplainer(class_names=self.__class_names)
        explanation = explainer.explain_instance(
            input_text,
            transformer.predict_proba,
            labels=[i for i in range(0,16)],
            num_samples=100,
            num_features=10
        )

        if device == 0:
            transformer.to_gpu()

        prediction = transformer.predict(input_text)

        if show:
            explanation.show_in_notebook(labels=[prediction])


    def print_parameters(self):
        parameters = f'''
            Parameter Name - Data Type - Explaination

            1. input_text - string - input text for the model
            2. transformer - Object - hugging face transformer for text classification
            3. show - Bool - Whether to show lime explaination or not
        '''

        print(parameters)




