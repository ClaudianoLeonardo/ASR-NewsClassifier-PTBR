import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
from transformers import Trainer, TrainingArguments


def load_data(path):
    """
    Carrega os dados de um arquivo CSV.

    Parameters:
    - path (str): O caminho para o arquivo CSV.

    Returns:
    - pd.DataFrame: Um DataFrame contendo os dados carregados.
    """
    df = pd.read_csv(path)
    return df


def limpar_texto(texto):
    """
    Limpa o texto removendo caracteres indesejados.

    Parameters:
    - texto (str): O texto a ser limpo.

    Returns:
    - str: O texto limpo.
    """
    texto_limpo = texto.replace('\n', ' ').replace('\t', ' ')
    texto_limpo = ' '.join(texto_limpo.split())
    return texto_limpo


def preprocessing_data(data):
    """
    Realiza pré-processamento nos dados.

    Parameters:
    - data (pd.DataFrame): O DataFrame contendo os dados a serem pré-processados.

    Returns:
    - pd.DataFrame: DataFrame pré-processado.
    """
    data = data[['conteudo_noticia', 'assunto']]
    data['texto_limpo'] = data['conteudo_noticia'].astype(str).apply(limpar_texto)
    data = data.rename(columns={'texto_limpo': 'texto', 'assunto': 'categoria'})
    data_pre_processed = data[['categoria', 'texto']]
    return data_pre_processed


def prepare_dataset(data):
    """
    Prepara o conjunto de dados para treinamento.

    Parameters:
    - data (pd.DataFrame): O DataFrame contendo os dados.

    Returns:
    - pd.DataFrame, pd.DataFrame: Conjuntos de treinamento e validação.
    """
    data['label'] = data['categoria'].astype('category').cat.codes
    data_train = data[['label', 'texto']]
    labels_encoding = pd.get_dummies(data_train['label']).astype('float')
    data_train['one_hot'] = labels_encoding.apply(lambda row: row.tolist(), axis=1)
    data_train_encoded = data_train[['one_hot', 'texto']]
    train_data, val_data = train_test_split(data_train_encoded, test_size=0.2, shuffle=True)

    return train_data, val_data


def modeling_dataset(data, column_mapping):
    """
    Modela o conjunto de dados para o Hugging Face.

    Parameters:
    - data (pd.DataFrame): O DataFrame contendo os dados.
    - column_mapping (dict): Mapeamento das colunas.

    Returns:
    - Dataset: Conjunto de dados do Hugging Face.
    """
    # Renomear as colunas conforme o mapeamento
    data = data.rename(columns=column_mapping)

    # Converter DataFrame para dicionário
    data_dict = data.to_dict(orient='list')

    # Criar dataset do Hugging Face
    hf_dataset = Dataset.from_dict(data_dict)

    return hf_dataset


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    """
    Função de tokenização para o conjunto de dados.

    Parameters:
    - examples (dict): Dicionário de exemplos.

    Returns:
    - dict: Exemplos tokenizados.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def encoder_dataset(dataset):
    """
    Codifica o conjunto de dados usando a função de tokenização.

    Parameters:
    - dataset (Dataset): Conjunto de dados a ser codificado.

    Returns:
    - None
    """
    encoded_dataset = dataset.map(tokenize_function, batched=True)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    Calcula métricas de avaliação para classificação multi-rótulo.

    Parameters:
    - predictions (list): Lista de previsões.
    - labels (list): Lista de rótulos reais.
    - threshold (float): Limiar para considerar a previsão como positiva.

    Returns:
    - dict: Métricas calculadas.
    """
   # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    """
    Calcula métricas a partir das previsões e rótulos reais.

    Parameters:
    - p (EvalPrediction): Objeto contendo previsões e rótulos.

    Returns:
    - dict: Métricas calculadas.
    """
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def train(model, encoded_train, encoded_val, args):
    """
    Treina o modelo usando o conjunto de treinamento e avaliação.

    Parameters:
    - model: Modelo a ser treinado.
    - encoded_train (Dataset): Conjunto de treinamento codificado.
    - encoded_val (Dataset): Conjunto de avaliação codificado.
    - args (TrainingArguments): Argumentos de treinamento.

    Returns:
    - Trainer: Objeto Trainer configurado para treinamento.
    """
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)
    return trainer


def inference(text, id2label, trainer):
    """
    Realiza inferência para um texto dado um modelo treinado.

    Parameters:
    - text (str): Texto para inferência.
    - id2label (dict): Mapeamento de índice para rótulo.
    - trainer (Trainer): Objeto Trainer treinado.

    Returns:
    - list: Rótulos previstos.
    """
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
    outputs = trainer.model(**encoding)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    return predicted_labels
