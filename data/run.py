from zenml import pipeline, step
from fetch_data import fetch_data
from pre_processing import pre_process_data
from segregation import segregate_data

@step
def load_step():
    """
    Step to load raw data from a CSV file.

    Returns:
    - dict: Dictionary containing the loaded raw data.
    """
    data = fetch_data('data/Historico_de_materias.csv')
    return {"data": data}

@step
def clean_step(data):
    """
    Step to perform data cleaning.

    Parameters:
    - data (dict): Dictionary containing the raw data.

    Returns:
    - dict: Dictionary containing the cleaned data.
    """
    cleaned_data = pre_process_data(data["data"])
    return {"cleaned_data": cleaned_data}

@step
def segregation_step(cleaned_data):
    """
    Step to segregate data into training and validation sets.

    Parameters:
    - cleaned_data (dict): Dictionary containing the cleaned data.

    Returns:
    - dict: Dictionary containing the preprocessed data.
    """
    preprocessed_data = segregate_data(cleaned_data["cleaned_data"])
    return {"preprocessed_data": preprocessed_data}

@pipeline
def my_pipeline():
    """
    Main pipeline to load, clean, and segregate data.
    """
    load = load_step()
    clean = clean_step(load)
    preprocess_step(clean)

if __name__ == "__main__":
    my_pipeline()
