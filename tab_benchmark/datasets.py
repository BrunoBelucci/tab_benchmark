from pathlib import Path
from openml import datasets
import pandas as pd

datasets_characteristics_path = Path(__file__).parent / 'openml_tasks.csv'


def get_dataset(dataset_name_or_id):
    if dataset_name_or_id.isdigit():
        dataset = datasets.get_dataset(int(dataset_name_or_id))
        target_name = dataset.default_target_attribute
        n_classes = dataset.qualities['NumberOfClasses']
        if n_classes == 2:
            task_name = 'binary_classification'
        elif n_classes > 2:
            task_name = 'classification'
        else:
            task_name = 'regression'
    else:
        datasets_characteristics = pd.read_csv(datasets_characteristics_path)
        dataset_characteristics = datasets_characteristics.loc[
            datasets_characteristics['dataset_name'] == dataset_name_or_id]
        dataset_id = dataset_characteristics['dataset_id'].values[0]
        task_name = dataset_characteristics['task_name'].values[0]
        target_name = dataset_characteristics['target_name'].values[0]
        n_classes = dataset_characteristics['n_classes'].values[0]
        dataset = datasets.get_dataset(int(dataset_id))
    return dataset, task_name, target_name, n_classes
