import evaluate
from transformers import Trainer, TrainingArguments
from development.datasets.osdg_dataset import OsdgDataset
from numpy import argmax


def compute_accuracy(eval_pred):
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

def fine_tune_transformer(model, tokenizer, tokenizer_args, data, dev_config):
    train_encodings = tokenizer(
        data['train'][0].tolist(),
        **tokenizer_args
    )
    valid_encodings = tokenizer(
        data['valid'][0].tolist(),
        padding=True,
        truncation=True
    )
    train_dataset = OsdgDataset(
        train_encodings, 
        data['train'][1],
    )
    valid_dataset = OsdgDataset(
        valid_encodings,
        data['valid'][1],
    )

    trainer_args = dev_config['trainer_args']

    training_args = TrainingArguments(
        output_dir=trainer_args['output_dir'],
        overwrite_output_dir=trainer_args['overwrite_output_dir'],
        num_train_epochs=trainer_args['num_train_epochs'],
        per_device_train_batch_size=trainer_args['per_device_train_batch_size'],
        per_device_eval_batch_size=trainer_args['per_device_eval_batch_size'],
        warmup_steps=trainer_args['warmup_steps'],
        weight_decay=trainer_args['weight_decay'],
        logging_dir=trainer_args['logging_dir'],
        logging_steps=trainer_args['logging_steps'],
        evaluation_strategy=trainer_args['evaluation_strategy']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_accuracy
    )

    result = trainer.train()

    return result