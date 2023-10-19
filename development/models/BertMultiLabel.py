from .MultilabelTextClassifier import MultilabelTextClassifier
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)

class BertMultiLabel(MultilabelTextClassifier):
    """ A class to load Bert model for multi-label classification using the HuggingFace API, the class inherits from the TextClassifier class which includes the model's functionalities such as running inference, see TextClassifier for more information.

    The configuration of the model, such as dorputs, can be set from this class.

    The configuration of the tokenizer, such as padding, can be set from this class.

    Attributes:
    -----------
    path: str
        Model path, the default path loads the pre-trained model from the HuggingFace Model-Hub
    
    """
    def __init__(self, path='bert-base-uncased'):
        names = [
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
        id2label = {idx:label for idx, label in enumerate(names)}
        label2id = {label:idx for idx, label in enumerate(names)}
        
        self.__model = BertForSequenceClassification.from_pretrained(
            path,
            problem_type='multi_label_classification',
            id2label=id2label,
            label2id=label2id,
            num_labels=16,
            output_attentions=True, 
            output_hidden_states=True,
            attention_probs_dropout_prob=3e-1,
            hidden_dropout_prob=3e-1,
            classifier_dropout=3e-1
        )
        self.__tokenizer  = BertTokenizerFast.from_pretrained(
            'bert-base-uncased', do_lower_case=True, padding_side='right'
        )
        self.__tokenizer_args = {
            'padding': 'max_length',
            'max_length': 512,
            'truncation': 'longest_first',
            'add_special_tokens': True, 
            'return_attention_mask': True,
            'return_tensors': 'pt'
        }
        super().__init__(self.__model, self.__tokenizer, self.__tokenizer_args)
