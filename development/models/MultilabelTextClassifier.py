class MultilabelTextClassifier:
    def __init__(self, model, tokenizer, tokenizer_args):
        self.__model = model
        self.__tokenizer  = tokenizer
        self.__tokenizer_args = tokenizer_args

    @property
    def model(self):
        return self.__model
    
    @property
    def tokenizer(self):
        return self.__tokenizer
    
    @property
    def tokenizer_args(self):
        return self.__tokenizer_args