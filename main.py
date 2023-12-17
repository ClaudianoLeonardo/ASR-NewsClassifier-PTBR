batch_size = 8
metric_name = "f1"
args = TrainingArguments(
    f"bert-finetuned-sem_eval-portuguese",
    evaluation_strategy = "steps",
    save_strategy = "steps",
    warmup_steps=5,
    logging_steps=100,
    save_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)


id2label = {0:'economia', 1:'esportes', 2:'famosos', 3:'politica', 4:'tecnologia'}
label2id = {'economia':0, 'esportes':1, 'famosos':2, 'politica':3, 'tecnologia':4}


column_mapping = {
    'one_hot': 'label',
    'texto': 'text'  
}


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=5,
                                                           id2label=id2label,
                                                           label2id=label2id)