from .TextClassificationModel import TextClassificationModel
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)

class Bert(TextClassificationModel):
    def __init__(self, path='bert-base-uncased'):
        self.__model = BertForSequenceClassification.from_pretrained(
            path,
            num_labels=16,
            output_attentions=False, 
            output_hidden_states=True
        )
        self.__tokenizer  = BertTokenizerFast.from_pretrained(
            'bert-base-uncased', do_lower_case=True
        )
        self.__tokenizer_args = {
            'truncation': True,
            'add_special_tokens': True, 
            'max_length': 216,
            'pad_to_max_length': True, 
            'return_attention_mask': True,
            'return_tensors': 'pt'
        }

    @property
    def model(self):
        return self.__model
    
    @property
    def tokenizer(self):
        return self.__tokenizer
    
    @property
    def tokenizer_args(self):
        return self.__tokenizer_args