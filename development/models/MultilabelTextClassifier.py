import numpy as np
from transformers import TextClassificationPipeline
from sklearn.metrics import classification_report
from torch import inference_mode
from torch.nn.functional import sigmoid
from ..utils import parse_sdg_label

class MultilabelTextClassifier:
    """ TextClassifier class provides functionalities to the multi-label models provided in the models module. Functionalities include running inference, evaluating the model on a set of data, and parsing the predictions.

    Attributes:
    -----------
    model (property): Object
        A Huggingface model
    tokenizer (property): Object
        A Huggingface tokenizer
    tokenizer_args (property):
        A dictionary holding the configuration of the tokenizer

    Methods:
    --------
    evaluate: 
        Evaluates a single-label classifier on a set of single-labeled data and returns a classification report
    evalaute_single_label:
        Evaluates a multi-label classifier on a set of single-labeled data and returns an accuracy score
    predict:
        Runs inference on single data point
    predict_batch:
        Runs inference on a batch of data
    cls_pipeline:
        Runs inference using Huggingface's TextClassificationPipeline
    predict_longtext:
        Runs inference on a single data point that exccedes a model's max token length using the window method
    predict_proba:
        Runs inference on a single data point or a batch of data and returns the probabilities of the predictions, used for interpretability packges (Lime)
    to_gpu:
        Loads the model to a GPU using the GPU ID
    parse_predictions:
        Parses predictions ran using Huggingface's TextClassificationPipeling

    """
     
    def __init__(self, model, tokenizer, tokenizer_args):
        self.__model = model
        self.__tokenizer  = tokenizer
        self.__tokenizer_args = tokenizer_args

    def evaluate(self, data, threshold=0.6):
        predictions = self.predict_batch(data[0])
        y_pred = np.where(predictions > threshold, 1, 0)
        y_true = data[1]
        evaluation = classification_report(y_true, y_pred, zero_division=1)

        return evaluation
    
    def evaluate_single_label(
        self,
        data,
        threshold=0.6,
        mode=['exact_match', 'included']
    ):
        predictions = self.cls_pipeline(data[0])
        y_true = data[1]

        if mode == 'exact_match':
            y_pred = self.parse_predictions(predictions, top_k=1)
            evaluation = classification_report(y_true, y_pred)

            return evaluation

        if mode == 'included':
            y_pred = self.parse_predictions(
                predictions,
                top_k=16,
                threshold=threshold
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
        
    def predict(self, text, device=0):
        tokens = self.__tokenizer.encode_plus(
            text,
            **self.__tokenizer_args
        ).to(device)

        input_dict = {
            'input_ids': tokens['input_ids'].long(),
            'attention_mask': tokens['attention_mask'].int()
        }
        self.__model.to(device)

        with inference_mode():
            prediction = self.__model(**input_dict)
            prediction = sigmoid(prediction[0])

        return prediction

    def predict_batch(self, texts):
        preidctions = np.zeros(shape=(len(texts), 16))

        for i, text in enumerate(texts):
            prediction = self.predict(text)
            preidctions[i] = prediction.cpu()

        return preidctions

    def predict_longtext(self):
        """ Implement the function
        """
        pass
    
    def cls_pipeline(self, data, device=0, parse=False, top_k=1, threshold=0.6):
        pipe = TextClassificationPipeline(
            model=self.__model,
            tokenizer=self.__tokenizer,
            top_k=None,
            device=device,
            max_length=512,
            truncation=True
        )

        if isinstance(data, str):
            predictions = pipe(data)

        else:
            predictions = pipe(data.tolist())


        if parse:
            predictions = self.parse_predictions(predictions, top_k, threshold)

        return predictions
    
    def to_gpu(self, gpu_id=0):
        self.__model == self.__model.to(gpu_id)
    
    def parse_predictions(self, predictions, top_k=1, threshold=0.6):
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