import evaluate
from transformers import Trainer, TrainingArguments
from development.datasets.TextDataset import TextDataset
from numpy import argmax

def compute_accuracy(eval_pred):
    """ A function to compute accuracy score, function can be passed to trainers to provide and evaluation during training and fine-tunning.

    Parameters:
    -----------
    eval_pred: list
        A list containing the logits and ground-truth labels

    Returns:
    --------
    accuracy: flaot
        Accuracy score
    """
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = argmax(logits, axis=-1)

    accuracy = metric.compute(predictions=predictions, references=labels)

    return accuracy

def fine_tune_transformer(model, tokenizer, tokenizer_args, data, dev_config):
    """" A function used to train or fine-tune a Huggingface model using the The Huggingface Transformers' libarary Trainer

    Parameters:
    -----------
    model: Object
        A Higgingface model (Bert, Albert, etc.)
    tokenizer: Object
        A Huggingface model tokenizer
    tokenizer_args: Object
        A dictionary containing the tokenizer's arguemnts
    data: List
        A list containing the training and validation  data
    dev_config: Dictionary
        The project uses a config file (config.js) containing the training arguments

    Returns:
    --------
    results: object
        The returned object from Huggingface's Trainer object
    
    """
    train_encodings = tokenizer(
        data['train'][0].tolist(),
        **tokenizer_args
    )
    valid_encodings = tokenizer(
        data['valid'][0].tolist(),
        **tokenizer_args
    )
    train_dataset = TextDataset(
        train_encodings, 
        data['train'][1],
    )
    valid_dataset = TextDataset(
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
