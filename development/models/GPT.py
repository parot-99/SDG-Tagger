from .TextClassificationModel import TextClassificationModel
from transformers import OpenAIGPTTokenizerFast, OpenAIGPTModel

class GPT(TextClassificationModel):
    def __init__(self, path='openai-gpt'):
        self.__model = OpenAIGPTModel.from_pretrained(
            path, num_labels=16
        )
        self.__tokenizer  = OpenAIGPTTokenizerFast.from_pretrained(
            'openai-gpt', do_lower_case=True, pad_token='#'
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