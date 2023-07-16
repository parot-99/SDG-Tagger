import numpy as np
from transformers import TextClassificationPipeline
from sklearn.metrics import classification_report
from torch import Tensor, cat, stack
from torch.nn.functional import softmax

class TextClassificationModel:
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

    def predict(self, text):
        tokens = self.__tokenizer.encode_plus(
            text,
            **self.__tokenizer_args
        )

        input_dict = {
            'input_ids': tokens['input_ids'].long(),
            'attention_mask': tokens['attention_mask'].int()
        }

        prediction = self.__model(**input_dict)
        prediction = softmax(prediction[0], dim=-1)
        prediction.mean(dim=0)
        prediction = prediction.argmax().item()

        return prediction
    
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



    
    def __parse_predictions(self, predictions):
        y_pred = np.zeros(len(predictions))

        for i, pred in enumerate(predictions):
            y_pred[i] = int(pred[0]['label'].split('_')[1])

        y_pred.astype(np.int64)

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
