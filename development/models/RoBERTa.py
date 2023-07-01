
from transformers import RobertaForSequenceClassification, AutoTokenizer
from .TextClassificationModel import TextClassificationModel


class RoBERTa(TextClassificationModel):
    def __init__(self, path='roberta-base'):      
        self.__model = RobertaForSequenceClassification.from_pretrained(
            path, num_labels=16
        )
        self.__tokenizer  = AutoTokenizer.from_pretrained(
            path, do_lower_case=True
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
