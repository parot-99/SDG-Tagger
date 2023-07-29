from .MultilabelTextClassifier import MultilabelTextClassifier
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)

class BertMultiLabel(MultilabelTextClassifier):
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
            output_attentions=False, 
            output_hidden_states=True,
            attention_probs_dropout_prob=1e-1,
            hidden_dropout_prob=3e-1,
            classifier_dropout=1e-1
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
