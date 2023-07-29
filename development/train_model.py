import evaluate
from transformers import Trainer, TrainingArguments
from development.datasets.OsdgDataset import OsdgDataset
from development.datasets.RelxDataset import RelxDataset
from numpy import argmax

def compute_accuracy(eval_pred):
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

def fine_tune_transformer(
    model,
    tokenizer,
    tokenizer_args,
    data,
    dataset,
    dev_config
):
    
    dataset = OsdgDataset if dataset == 'osdg' else RelxDataset
    train_encodings = tokenizer(
        data['train'][0].tolist(),
        **tokenizer_args
    )
    valid_encodings = tokenizer(
        data['valid'][0].tolist(),
        **tokenizer_args
    )
    train_dataset = dataset(
        train_encodings, 
        data['train'][1],
    )
    valid_dataset = dataset(
        valid_encodings,
        data['valid'][1],
    )

    trainer_args = TrainingArguments(**dev_config['training_args'])
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # compute_metrics=compute_accuracy
    )

    result = trainer.train()
    trainer.save_model(output_dir=dev_config['training_args']['output_dir'])

    return result
