from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    TextClassificationPipeline
)

class SDGsClassifier:
    def __init__(self, multilabel=True, path='', device=0) -> None:
        self.__device = device
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
            num_labels=16,
            id2label=id2label,
            label2id=label2id,
            output_attentions=True, 
            output_hidden_states=True,
            attention_probs_dropout_prob=1e-1,
            hidden_dropout_prob=1e-1,
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
        self.__classification_pipe = TextClassificationPipeline(
            model=self.__model,
            tokenizer=self.__tokenizer,
            top_k=None,
            device=self.__device,
            max_length=512,
            truncation=True
        )

    def classify_text(self, text):
        prediction = self.__classification_pipe(text)[0]

        return prediction

    def classify_texts(self):
        pass

    def parse_predictions(self):
        pass