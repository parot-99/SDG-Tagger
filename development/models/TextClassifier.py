import numpy as np
from transformers import TextClassificationPipeline
from sklearn.metrics import classification_report
from torch import Tensor, cat, stack, inference_mode
from torch.nn.functional import softmax
from ..utils import parse_sdg_label

class TextClassifier:
    """ TextClassifier class provides functionalities to the single-label models provided in the models module. Functionalities include running inference, evaluating the model on a set of data, and parsing the predictions.

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
    evaluate_multilabel:
        Evaluates a single-label classifier on a set of multi-labeled data and returns an accuracy score
    predict:
        Runs inference on single data point
    predict_batch:
        Runs inference on a batch of data
    predict_longtext:
        Runs inference on a single data point that exccedes a model's max token length using the window method
    cls_pipeline:
        Runs inference using Huggingface's TextClassificationPipeline
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

    def evaluate(self, data, add_one=False):
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
        y_pred = self.__parse_predictions(predictions)

        if add_one:
            y_pred += 1

        evaluation = classification_report(y_true, y_pred)

        return evaluation
    
    def evaluate_multilabel(self, data, add_one=False):
        """ The function evaluates a single-label model on multi-labled data using a hit-rate. If the prediction is in the list of the multilabels of the relavant instance, the prediction is considered as a hit. An accuracy score is calculated using the previous method.

        Parameters:
        -----------
        data: numpy array
            A Numpy array containing data to predict
        add_one: bool
            Since Bert's prediction are 0-indexed, add one adds 1 to each preidction to make it corespond to SDG. For example, if Bert predicts 3, it means the prediction is SDG 4 Quality of Education, if add one is true, the prediction will be 4 rather than 3.

        Returns:
        --------
        accuracy: int
            accuracy score of evaluating the model on the provided data list
        """
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
        y_pred = self.__parse_predictions(predictions)

        if add_one:
            y_pred += 1

        data_length = len(y_true)
        accuracy_list = []

        for i in range(data_length):
            prediction = y_pred[i]

            if prediction in y_true[i]:
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
            prediction = softmax(prediction[0], dim=-1)
            # prediction.mean(dim=0)
            prediction = prediction.argmax().item()

        return prediction
    
    def predict_batch(self, texts):
        preidctions = np.zeros(shape=(len(texts), 16))

        for i, text in enumerate(texts):
            prediction = self.predict(text)
            preidctions[i] = prediction.cpu()

        return preidctions
    
    def cls_pipeline(self, data, device=0, parse=False):
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
            predictions = self.parse_predictions(predictions)

        return predictions

    def predict_long_text(self, text):
        window_size = 510
        cls_token = [101]
        sep_token = [102]
        tokens = self.__tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_tensors='pt'
        )

        input_id_chunks = list(tokens['input_ids'][0].split(window_size))
        attention_mask_chunks = list(
            tokens['attention_mask'][0].split(window_size)
        )

        for i in range(len(input_id_chunks)):
            input_id_chunks[i] = cat([
                Tensor(cls_token),
                input_id_chunks[i],
                Tensor(sep_token),
            ])
            attention_mask_chunks[i] = cat([
                Tensor([1]),
                attention_mask_chunks[i],
                Tensor([1])
            ])

            pad_length = window_size + 2 - input_id_chunks[i].shape[0]

            if pad_length > 0:
                input_id_chunks[i] = cat([
                    input_id_chunks[i], Tensor([0] * pad_length)
                ])
                attention_mask_chunks[i] = cat([
                    attention_mask_chunks[i], Tensor([0] * pad_length)
                ])

        inpuut_ids = stack(input_id_chunks)
        attention_mask = stack(attention_mask_chunks)

        input_dict = {
            'input_ids': inpuut_ids.long(),
            'attention_mask': attention_mask.int()
        }

        prediction = self.__model(**input_dict)
        prediction = softmax(prediction[0], dim=-1)
        prediction.mean(dim=0)
        prediction = prediction.argmax().item()

        return prediction

    def predict_proba(self, text):
        model_inputs = self.__tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(0)
        self.__model.to(0)
        outputs = self.__model(**model_inputs)
        logits = outputs.logits
        probas = softmax(logits, dim=-1).detach().cpu().numpy()

        return probas
   
    def to_gpu(self, gpu_id=0):
        self.__model.to(gpu_id)

    def __parse_predictions(self, predictions):
        y_pred = np.zeros(len(predictions), dtype=np.int64)

        for i, pred in enumerate(predictions):
            y_pred[i] = parse_sdg_label(pred[0]['label'], training=True)

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
