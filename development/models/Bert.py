from .TextClassifier import TextClassifier
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)

class Bert(TextClassifier):
    def __init__(self, path='bert-base-uncased'):
        self.__model = BertForSequenceClassification.from_pretrained(
            path,
            num_labels=16,
            output_attentions=True, 
            output_hidden_states=True,
            attention_probs_dropout_prob=1e-1,
            hidden_dropout_prob=3e-1,
            classifier_dropout=None
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
