
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizerFast
from .TextClassificationModel import TextClassificationModel


class RoBERTa(TextClassificationModel):
    def __init__(self, path='roberta-base'):      
        self.__model = RobertaForSequenceClassification.from_pretrained(
            path, num_labels=16
        )
        self.__tokenizer  = RobertaTokenizerFast.from_pretrained(
            'roberta-base', do_lower_case=True
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
