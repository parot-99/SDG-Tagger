import numpy as np
from transformers import TextClassificationPipeline
from sklearn.metrics import classification_report
from torch import Tensor, cat, stack, inference_mode
from torch.nn.functional import softmax, sigmoid
from pandas import DataFrame
from ..utils import parse_sdg_label

class MultilabelTextClassifier:
    def __init__(self, model, tokenizer, tokenizer_args):
        self.__model = model
        self.__tokenizer  = tokenizer
        self.__tokenizer_args = tokenizer_args

    def evaluate(self, data):
        predictions = self.predict_batch(data[0])
        y_pred = np.where(predictions > 0.5, 1, 0)
        y_true = data[1]
        evaluation = classification_report(y_true, y_pred, zero_division=1)
        # evaluation = classification_report(y_true, y_pred, output_dict=True)
        # evaluation = DataFrame(evaluation).T

        return evaluation
    
    def evaluate_single_label(self, data, mode=['exact_match', 'included']):
        pipe = TextClassificationPipeline(
            model=self.__model,
            tokenizer=self.__tokenizer,
            top_k=None,
            device=0,
            max_length=512,
            truncation=True
        )
        y_true = data[1]
        predictions = pipe(data[0].tolist())

        if mode == 'exact_match':
            y_pred = self.parse_predictions(predictions, top_k=1)
            evaluation = classification_report(y_true, y_pred)

            return evaluation

        if mode == 'included':
            y_pred = self.parse_predictions(
                predictions,
                top_k=16,
                threshold=0.6
            )
            data_length = len(predictions)
            accuracy_list = []

            for i in range(data_length):
                if y_true[i] in y_pred[i]:
                    accuracy_list.append(1)

                else:
                    accuracy_list.append(0)

            ones = accuracy_list.count(1)
            zeros = accuracy_list.count(0)
            accuracy = ones / (ones + zeros)

            return accuracy
        
    def predict(self, text):
        pipe = TextClassificationPipeline(
            model=self.__model,
            tokenizer=self.__tokenizer,
            top_k=None,
            device=0,
            max_length=512,
            truncation=True
        )
        prediction = pipe(text)

        return prediction

    def predict_batch(self, texts):
        preidctions = np.zeros(shape=(len(texts), 16))

        for i, text in enumerate(texts):
            tokens = self.__tokenizer.encode_plus(
                text,
                **self.__tokenizer_args
            ).to(0)

            input_dict = {
                'input_ids': tokens['input_ids'].long(),
                'attention_mask': tokens['attention_mask'].int()
            }

            with inference_mode():
                prediction = self.__model(**input_dict)
                prediction = sigmoid(prediction[0])
                preidctions[i] = prediction.cpu()

        return preidctions
    
    def to_gpu(self, gpu_id=0):
        self.__model == self.__model.to(gpu_id)
    
    def parse_predictions(self, predictions, top_k=1, threshold=0.0):
        if top_k == 1:
            y_pred = np.zeros(len(predictions))

            for i, pred in enumerate(predictions):
                y_pred[i] = parse_sdg_label(pred[0]['label'])

            y_pred.astype(np.int64)

            return y_pred
        
        if top_k > 1 and top_k < 17:
            y_pred = []

            for i, pred_list in enumerate(predictions):
                y_pred.append([])

                for j in range(top_k):
                    if pred_list[j]['score'] > threshold:
                        y_pred[i].append(
                            parse_sdg_label(pred_list[j]['label'])
                        )

            return y_pred

    @property
    def model(self):
        return self.__model
    
    @property
    def tokenizer(self):
        return self.__tokenizer
    
    @property
    def tokenizer_args(self):
        return self.__tokenizer_args