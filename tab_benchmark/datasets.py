from pathlib import Path
from openml import datasets
import pandas as pd

datasets_characteristics_path = Path(__file__).parent / 'datasets_characteristics.csv'


def get_dataset(dataset_name_or_id):
    if dataset_name_or_id.isdigit():
        dataset = datasets.get_dataset(int(dataset_name_or_id))
        target = dataset.default_target_attribute
        if dataset.retrieve_class_labels(target) is None:
            task = 'regression'
        else:
            task = 'classification'
    else:
        datasets_characteristics = pd.read_csv(datasets_characteristics_path)
        dataset_characteristics = datasets_characteristics.loc[datasets_characteristics['name'] == dataset_name_or_id]
        dataset_id = dataset_characteristics['id'].values[0]
        task = dataset_characteristics['task'].values[0]
        target = dataset_characteristics['target'].values[0]
        dataset = datasets.get_dataset(dataset_id)
    return dataset, task, target
