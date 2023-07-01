import numpy as np
from transformers import TextClassificationPipeline
from sklearn.metrics import classification_report

class TextClassificationModel:
    def __init__(self):
        pass

    def evaluate(self, model, tokenizer, data, add_one=False):
        pipe = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            device=0,
            max_length=512,
            truncation=True
        )

        y_true = data[1]
        predictions = pipe(data[0].tolist())
        y_pred = np.zeros(len(predictions))

        for i, pred in enumerate(predictions):
            y_pred[i] = int(pred[0]['label'].split('_')[1])

        y_pred.astype(np.int64)

        if add_one:
            y_pred += 1

        evaluation = classification_report(y_true, y_pred)

        return evaluation

    def predict(self, model, tokenizer, data):
        pipe = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            device=0,
            max_length=512,
            truncation=True
        )

        predictions = pipe(data)

        return predictions
    
    def parse_predictions(self, predictions):
        y_pred = np.zeros(len(predictions))

        for i, pred in enumerate(predictions):
            y_pred[i] = int(pred[0]['label'].split('_')[1])

        y_pred.astype(np.int64)

        return y_pred
