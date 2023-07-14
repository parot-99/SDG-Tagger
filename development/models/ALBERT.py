from .TextClassificationModel import TextClassificationModel
from transformers import (
    AlbertTokenizerFast,
    AlbertForSequenceClassification
)

class ALBERT(TextClassificationModel):
    def __init__(self, path='albert-base-v2'):
        self.__model = AlbertForSequenceClassification.from_pretrained(
            path, num_labels=16
        )
        self.__tokenizer  = AlbertTokenizerFast.from_pretrained(
            'albert-base-v2', do_lower_case=True
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
        