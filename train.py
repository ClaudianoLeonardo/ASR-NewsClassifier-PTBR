from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import mlflow
from utils import compute_metrics, tokenize_function

def train(model_id, num_labels, batch_size, metric_name, model_name, train_dataset, val_dataset, num_train_epochs):
    """
    Train a sequence classification model.

    Parameters:
    - model_id (str): The identifier of the pre-trained model to be used.
    - num_labels (int): The number of labels for classification.
    - batch_size (int): The batch size for training.
    - metric_name (str): The metric to be used for model evaluation.
    - model_name (str): The name to be used for saving the trained model.
    - train_dataset (Dataset): Training dataset.
    - val_dataset (Dataset): Validation dataset.
    - num_train_epochs (int): The number of training epochs.

    Returns:
    - Trainer: Trained Trainer object.
    """
    id2label = {0: 'economia', 1: 'esportes', 2: 'famosos', 3: 'politica', 4: 'tecnologia'}
    label2id = {'economia': 0, 'esportes': 1, 'famosos': 2, 'politica': 3, 'tecnologia': 4}

    # Tokenize the datasets
    encoded_train = train_dataset.map(tokenize_function, batched=True)
    encoded_val = val_dataset.map(tokenize_function, batched=True)

    # Load the pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        problem_type="multi_label_classification",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    args = TrainingArguments(
        model_name,
        evaluation_strategy="steps",
        save_strategy="steps",
        warmup_steps=5,
        logging_steps=100,
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name
    )

    # Trainer initialization
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        compute_metrics=compute_metrics
    )

    # Training and logging with mlflow
    with mlflow.start_run():
        trainer.train()
        mlflow.log_params(args.to_dict())
        final_metrics = trainer.evaluate(encoded_val)
        mlflow.log_metric("final_f1_micro", final_metrics["eval_f1"])
        mlflow.log_metric("final_roc_auc", final_metrics["eval_roc_auc"])
        mlflow.log_metric("final_accuracy", final_metrics["eval_accuracy"])
        mlflow.log_metric("final_runtime", final_metrics["eval_runtime"])
    mlflow.end_run()

    return trainer

